#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py  (monthly IO + multi-head)
----------------------------------------
- Reads **monthly** Parquet files:
    features/exchange={exchange}/symbol={symbol}/year=YYYY/month=MM/features-YYYY-MM.parquet
    labels  /exchange={exchange}/symbol={symbol}/year=YYYY/month=MM/features-YYYY-MM.parquet  # (per user's current layout)
    (For robustness, labels will also try labels-YYYY-MM.parquet if the above is absent.)
- Inner-joins by (symbol, open_time).
- Builds sliding windows (seq_len) and trains an LSTM backbone with either:
    * classification head, or
    * regression head, or
    * **multi-head** (both at once, shared backbone).
- Early stopping, cosine LR, AMP mixed precision, gradient clipping.
- Saves best checkpoint + standardization stats + feature list (+ class map if cls involved).

Feature Categories (由 feature_builder.py 產生):
----------------------------------------
1. Base Features (基礎特徵):
    - symbol: 交易對
    - open_time: 時間戳
    - close: 收盤價
    - volume: 成交量 (若可用)
    - number_of_trades: 交易次數 (若可用)

2. Return Features (報酬特徵):
    - ret_1m: 1分鐘對數報酬率 (log return)

3. Technical Indicators (技術指標):
    - sma_5/10/20/50/100/200: 簡單移動平均線
    - rsi_14: 14期相對強弱指標 (Wilder)
    - atr_14: 14期真實波動範圍 (Wilder)

4. Volatility Features (波動率特徵):
    - rv_5/15/30/60/120/240: 已實現波動率
    (基於1分鐘對數報酬計算，按不同時間窗口)

5. Volume Analysis (成交量分析):
    - vol_z_30/60/120: 成交量 Z-Score (30/60/120 分鐘窗口)
    - trades_z_30/60/120: 交易次數 Z-Score (30/60/120 分鐘窗口)

Labels (由 label_builder.py 產生):
--------------------------------
1. Return-based Labels (報酬類標籤):
    - y_reg_ret_Hm: H分鐘未來對數報酬
    - y_cls_sign_Hm: 價格方向標籤 (-1/0/1，考慮死區 epsilon)

2. Trading Signal Labels (交易訊號標籤):
    - y_tp_sl_Hm: 止盈/止損先觸發標籤 (+1/-1/0)
    - y_tp_sl_Hm_t_hit: 觸發時間點

3. Triple Barrier Labels (三重障礙標籤):
    - y_tb_Hm: 三重障礙標籤 (+1/-1/0)
    - y_tb_Hm_t_hit: 觸發時間點

4. Volatility Labels (波動率標籤):
    - y_vol_Hm: H分鐘未來實現波動率

註: H 為預測視窗 (horizon)，預設為 60 分鐘
"""
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import sys
import json
from tqdm import tqdm

# ----------------------
# Time helpers
# ----------------------
def month_span(start: str, end: str):
    s = pd.to_datetime(start).to_pydatetime()
    e = pd.to_datetime(end).to_pydatetime()
    y, m = s.year, s.month
    while True:
        first = datetime(y, m, 1)
        nxt = datetime(y + (m==12), (m % 12) + 1, 1)
        if first < e:
            yield y, m
        y, m = (y + (m==12), (m % 12) + 1)
        if datetime(y, m, 1) >= e:
            break

# ----------------------
# IO
# ----------------------
def read_monthly(root: Path, exchange: str, symbol: str, year: int, month: int, kind: str, interval: int = 10) -> pd.DataFrame:
    """讀取每月的特徵或標籤檔案。

    Args:
        root: 根目錄
        exchange: 交易所
        symbol: 交易對
        year: 年份
        month: 月份
        kind: "features" 或 "labels"
        interval: 重採樣間隔（分鐘）

    Returns:
        DataFrame，若檔案不存在則為空 DataFrame
    """
    # 加入時間間隔資料夾
    fpath = (root / f"exchange={exchange}" / f"{interval}min" / f"symbol={symbol}" /
            f"year={year}" / f"month={month:02d}" / f"{kind}-{year}-{month:02d}.parquet")
    
    if not fpath.exists():
        # 相容舊路徑（不含時間間隔資料夾）
        fpath = (root / f"exchange={exchange}" / f"symbol={symbol}" /
                f"year={year}" / f"month={month:02d}" / f"{kind}-{year}-{month:02d}.parquet")
    
    if not fpath.exists():
        print(f"[warning] {fpath} not found.")
        return pd.DataFrame()
    
    try:
        return pd.read_parquet(fpath)
    except Exception as e:
        print(f"[error] Failed to read {fpath}: {e}")
        return pd.DataFrame()

def load_merged_months(features_root: Path, labels_root: Path, exchange: str, symbol: str,
                      start: str, end: str, interval: int = 10) -> pd.DataFrame:
    """讀取並合併多個月的特徵與標籤。

    Args:
        features_root: 特徵根目錄
        labels_root: 標籤根目錄
        exchange: 交易所
        symbol: 交易對
        start: 開始日期 (YYYY-MM-DD)
        end: 結束日期 (YYYY-MM-DD)
        interval: 重採樣間隔（分鐘）

    Returns:
        合併後的 DataFrame
    """
    fdfs, ldfs = [], []
    for y, m in month_span(start, end):
        fdf = read_monthly(features_root, exchange, symbol, y, m, "features", interval)
        ldf = read_monthly(labels_root,   exchange, symbol, y, m, "labels", interval)
        if not fdf.empty: fdfs.append(fdf)
        if not ldf.empty: ldfs.append(ldf)
    if not fdfs or not ldfs:
        return pd.DataFrame()

    fdf = pd.concat(fdfs, ignore_index=True)
    ldf = pd.concat(ldfs, ignore_index=True)

    fdf["open_time"] = pd.to_datetime(fdf["open_time"], utc=True)
    ldf["open_time"] = pd.to_datetime(ldf["open_time"], utc=True)
    s = pd.to_datetime(start, utc=True); e = pd.to_datetime(end, utc=True)
    fdf = fdf[(fdf["open_time"] >= s) & (fdf["open_time"] < e)]
    ldf = ldf[(ldf["open_time"] >= s) & (ldf["open_time"] < e)]

    keep_labels = [c for c in ldf.columns if c not in ("symbol","open_time","exchange","close","year","month")]
    merged = pd.merge(fdf, ldf[["symbol","open_time"] + keep_labels],
                      on=["symbol","open_time"], how="inner")
    merged = merged.sort_values("open_time").reset_index(drop=True)
    return merged

# ----------------------
# Features / sequences
# ----------------------
def infer_feature_columns(df: pd.DataFrame, exclude):
    """推斷特徵欄位，依據預定義的特徵類別進行分類與篩選。
    
    Args:
        df: 包含特徵的DataFrame
        exclude: 要排除的欄位清單
    
    Returns:
        dict: 按類別分類的特徵欄位字典
        list: 所有特徵欄位列表
    """
    # 基本要排除的欄位
    ignore = {"symbol", "open_time", "year", "month"} | set(exclude)
    
    # # 排除合併後的重複列（帶_x或_y後綴的列）
    # duplicate_suffixes = []
    # for col in df.columns:
    #     if col.endswith('_x') or col.endswith('_y'):
    #         base_name = col[:-2]
    #         if f"{base_name}_x" in df.columns and f"{base_name}_y" in df.columns:
    #             if col.endswith('_y'):  # 只保留 _x 版本
    #                 duplicate_suffixes.append(col)
                
    # 排除標籤相關列
    label_prefixes = ["y_cls_", "y_reg_", "y_tp_sl_", "y_tb_", "y_vol_"]
    
    # 更新需要排除的列
    # ignore = ignore | set(duplicate_suffixes)
    ignore.update([col for col in df.columns if any(col.startswith(prefix) for prefix in label_prefixes)])
    
    # 定義各類特徵的前綴或關鍵字 (依照 feature_builder.py 的實際輸出)
    feature_patterns = {
        "base": ["close", "volume", "number_of_trades"],
        "returns": ["ret_1m"],
        "technical": ["sma_", "rsi_14", "atr_14"],
        "volatility": ["rv_"],
        "volume_analysis": ["vol_z_", "trades_z_"]
    }
    
    # 按類別篩選特徵
    features_by_category = {cat: [] for cat in feature_patterns}
    other_features = []
    
    for col in df.columns:
        if col in ignore or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # 檢查該列是否屬於某個預定義類別
        categorized = False
        for cat, patterns in feature_patterns.items():
            if any(col.startswith(p) or col == p for p in patterns):
                features_by_category[cat].append(col)
                categorized = True
                break
                
        # 如果不屬於任何預定義類別，放入其他類別
        if not categorized:
            other_features.append(col)
    
    # 如果有其他未分類的特徵，加入到結果中
    if other_features:
        features_by_category["other"] = other_features
    
    # 返回分類結果和完整特徵列表
    all_features = [f for cat_features in features_by_category.values() for f in cat_features]
    
    # 調試輸出
    print("\n=== Feature Selection Debug ===")
    print("All columns:", df.columns.tolist())
    print("\nFeatures by category:")
    for cat, feats in features_by_category.items():
        print(f"{cat}: {feats}")
    print("\nTotal features selected:", len(all_features))
    print("Selected features:", all_features)
    print("===========================\n")
    
    return features_by_category, all_features

def make_sequences(df: pd.DataFrame, feats, targets: dict, seq_len: int):
    arr = df[feats].to_numpy(dtype=np.float32)
    n = len(df)
    X = []
    y_cls = []
    y_reg = []
    has_cls = targets.get("cls") is not None and targets["cls"] in df.columns
    has_reg = targets.get("reg") is not None and targets["reg"] in df.columns
    cls_vals = df[targets["cls"]].to_numpy() if has_cls else None
    reg_vals = df[targets["reg"]].to_numpy() if has_reg else None
    for i in range(n - seq_len + 1):
        X.append(arr[i:i+seq_len])
        y_cls.append(cls_vals[i+seq_len-1] if has_cls else np.nan)
        y_reg.append(reg_vals[i+seq_len-1] if has_reg else np.nan)
    X = np.stack(X) if X else np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)
    y_cls = np.array(y_cls)
    y_reg = np.array(y_reg, dtype=np.float32)
    return X, y_cls, y_reg

class SeqDataset(Dataset):
    def __init__(self, X, y_cls, y_reg, timestamps=None, weight_mode="none", half_life_days=90.0, task="auto"):
        self.X = torch.from_numpy(X).float()
        self.cls_label_map = None
        # map classification labels if they look categorical
        if np.issubdtype(y_cls.dtype, np.integer) or set(np.unique(y_cls[~pd.isna(y_cls)])).issubset({-1,0,1,2,3}):
            uniq = sorted(np.unique(y_cls[~pd.isna(y_cls)]))
            self.cls_label_map = {v:i for i,v in enumerate(uniq)}
            cls_idx = np.array([self.cls_label_map.get(v, -1) if not np.isnan(v) else -1 for v in y_cls], dtype=np.int64)
        else:
            cls_idx = np.full_like(y_cls, -1, dtype=np.int64)
        self.y_cls = torch.from_numpy(cls_idx)       # -1 = invalid
        self.y_reg = torch.from_numpy(y_reg).float() # NaN = invalid

        self.weights = None
        if weight_mode == "time" and timestamps is not None:
            # Compute time-based weights
            now = timestamps.max()
            age_days = (now - timestamps).total_seconds() / 86400
            weights = np.exp(-np.log(2) * age_days / half_life_days)
            # Normalize weights to mean=1
            self.weights = torch.from_numpy(weights / weights.mean()).float()
        
        if task == "auto":
            if (self.y_cls >= 0).sum() > 0 and torch.isfinite(self.y_reg).sum() > 0:
                self.task = "multi"
            elif (self.y_cls >= 0).sum() > 0:
                self.task = "cls"
            else:
                self.task = "reg"
        else:
            self.task = task

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        w = self.weights[idx] if self.weights is not None else 1.0
        return self.X[idx], self.y_cls[idx], self.y_reg[idx], w

class LazySeqDataset(Dataset):
    """
    動態產生序列，避免一次將所有序列展開到記憶體。
    """
    def __init__(self, df, feats, targets, seq_len, weight_mode="none", half_life_days=90.0, task="auto"):
        self.feats = feats
        self.targets = targets
        self.seq_len = seq_len
        self.symbol_arrays = {}    # 特徵陣列
        self.symbol_labels = {}    # 標籤陣列（新增）
        self.indices = []
        
        # 預先轉換所有資料為 numpy array
        for sym, dfg in df.groupby("symbol"):
            # 特徵轉 numpy
            feat_arr = dfg[feats].to_numpy(dtype=np.float32)
            self.symbol_arrays[sym] = feat_arr
            
            # 標籤轉 numpy（新增）
            label_dict = {}
            if targets.get("cls") in dfg.columns:
                label_dict["cls"] = dfg[targets["cls"]].to_numpy()
            if targets.get("reg") in dfg.columns:
                label_dict["reg"] = dfg[targets["reg"]].to_numpy()
            self.symbol_labels[sym] = label_dict
            
            # 產生可用的索引
            for i in range(len(dfg) - seq_len + 1):
                self.indices.append((sym, i))
        
        # 預先計算 label map（僅限分類）
        y_cls_all = []
        for sym, labels in self.symbol_labels.items():
            if "cls" in labels:
                y_cls_all.extend(labels["cls"][self.seq_len-1:])
        y_cls_all = np.array(y_cls_all)
        
        # 設定分類標籤對應
        self.cls_label_map = None
        if len(y_cls_all) > 0 and (
            np.issubdtype(y_cls_all.dtype, np.integer) or
            set(np.unique(y_cls_all[~pd.isna(y_cls_all)])).issubset({-1,0,1,2,3})
        ):
            uniq = sorted(np.unique(y_cls_all[~pd.isna(y_cls_all)]))
            self.cls_label_map = {v:i for i,v in enumerate(uniq)}
        
        self.weights = None
        if weight_mode == "time":
            # 確保時間是 UTC 時區
            if df["open_time"].dt.tz is None:
                df["open_time"] = df["open_time"].dt.tz_localize("UTC")
            now = df["open_time"].max()
            # Store timestamps for the last point of each sequence
            timestamps = []
            for sym, dfg in df.groupby("symbol"):
                ts = dfg["open_time"].values[self.seq_len-1:]
                timestamps.extend(ts)
            
            # 確保所有時間戳都有時區
            timestamps = pd.to_datetime(timestamps).tz_localize(None).tz_localize("UTC")
            age_days = [(now - t).total_seconds() / 86400 for t in timestamps]
            weights = np.exp(-np.log(2) * np.array(age_days) / half_life_days)
            # Normalize weights to mean=1
            self.weights = weights / weights.mean()

        self.task = task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sym, start_idx = self.indices[idx]
        
        # 取特徵序列
        X = self.symbol_arrays[sym][start_idx:start_idx+self.seq_len]
        
        # 取標籤（序列最後一個時間點）
        labels = self.symbol_labels[sym]
        label_idx = start_idx + self.seq_len - 1
        
        y_cls = labels["cls"][label_idx] if "cls" in labels else np.nan
        y_reg = labels["reg"][label_idx] if "reg" in labels else np.nan
        
        # 分類 label 映射
        if self.cls_label_map is not None and not np.isnan(y_cls):
            y_cls_idx = self.cls_label_map.get(y_cls, -1)
        else:
            y_cls_idx = -1
            
        w = self.weights[idx] if self.weights is not None else 1.0
        return (
            torch.from_numpy(X).float(),
            torch.tensor(y_cls_idx, dtype=torch.int64),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32)
        )

# ----------------------
# Model
# ----------------------
class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(self.norm(out[:, -1, :]))
        return h

class LSTMHeads(nn.Module):
    def __init__(self, backbone: LSTMBackbone, n_classes: int = 3):
        super().__init__()
        self.backbone = backbone
        H = backbone.norm.normalized_shape[0]
        self.head_cls = nn.Linear(H, n_classes)
        self.head_reg = nn.Linear(H, 1)
    def forward(self, x):
        h = self.backbone(x)
        return self.head_cls(h), self.head_reg(h).squeeze(-1)

# ----------------------
# Train / Eval
# ----------------------
def train_model(model, loaders, device, task="multi", epochs=50, patience=6, lr=1e-3, weight_decay=1e-4,
                alpha=0.5, ckpt="lstm.ckpt"):
    train_loader, val_loader = loaders
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(len(train_loader),1) * max(epochs,1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps) if total_steps>0 else None
    ce = nn.CrossEntropyLoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf"); best_epoch = -1
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0; ntr = 0
        for xb, yb_cls, yb_reg, wb in train_loader:  # Add weights to batch
            if torch.isnan(xb).any():
                print(f"[warning] NaN in features")
                continue

            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            yb_reg = yb_reg.to(device)
            wb = wb.to(device)  # Move weights to device
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type="cuda"):
                logit, pred_reg = model(xb)
                loss = 0.0
                
                if task in ("cls","multi"):
                    mask_cls = yb_cls >= 0
                    n_valid = mask_cls.sum().item()
                    if n_valid > 0:
                        loss_cls = (ce(logit[mask_cls], yb_cls[mask_cls]) * wb[mask_cls]).mean()  # Apply weights
                        if torch.isnan(loss_cls):
                            print(f"[warning] NaN in cls loss")
                            continue
                        loss = loss + (alpha if task=="multi" else 1.0) * loss_cls
                    else:
                        print(f"[warning] No valid classification labels in batch")
                        
                if task in ("reg","multi"):
                    mask_reg = torch.isfinite(yb_reg)
                    n_valid = mask_reg.sum().item()
                    if n_valid > 0:
                        loss_reg = (huber(pred_reg[mask_reg], yb_reg[mask_reg]) * wb[mask_reg]).mean()  # Apply weights
                        if torch.isnan(loss_reg):
                            print(f"[warning] NaN in reg loss")
                            continue
                        loss = loss + ((1-alpha) if task=="multi" else 1.0) * loss_reg
                    else:
                        print(f"[warning] No valid regression labels in batch")
                
                if loss == 0.0:
                    print(f"[warning] Zero loss (no valid labels)")
                    continue
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sch is not None:
                sch.step()
                
            tr_loss += loss.item() * xb.size(0)
            ntr += xb.size(0)
            
        tr_loss = tr_loss / max(ntr,1) if ntr > 0 else float('nan')
        
        # Validation loop (unchanged, no weights needed)
        model.eval()
        with torch.no_grad():
            va_loss = 0.0; nva = 0
            for xb, yb_cls, yb_reg, _ in val_loader:  # Ignore weights in validation
                xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
                logit, pred_reg = model(xb)
                loss = 0.0
                if task in ("cls","multi"):
                    mask_cls = yb_cls >= 0
                    if mask_cls.any():
                        loss += (alpha if task=="multi" else 1.0) * ce(logit[mask_cls], yb_cls[mask_cls]).mean()
                if task in ("reg","multi"):
                    mask_reg = torch.isfinite(yb_reg)
                    if mask_reg.any():
                        loss += ((1-alpha) if task=="multi" else 1.0) * huber(pred_reg[mask_reg], yb_reg[mask_reg]).mean()
                va_loss += loss.item() * xb.size(0); nva += xb.size(0)
            va_loss /= max(nva,1)

        # print(f"[epoch {ep}] train={tr_loss:.6f} val={va_loss:.6f} best={best_val:.6f}@{best_epoch}")
        # print(f"[stats] train samples={ntr}, valid samples={nva}")
        logging.info(f"[epoch {ep}] train={tr_loss:.6f} val={va_loss:.6f} best={best_val:.6f}@{best_epoch}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss; best_epoch = ep
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
        elif ep - best_epoch >= patience:
            print(f"[early-stop] best={best_val:.6f} @ epoch {best_epoch}")
            break

    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    return model

def evaluate(model, loader, device, task):
    model.eval()
    out = {}
    ce = nn.CrossEntropyLoss(reduction="sum")
    l1 = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        correct = 0; total = 0
        loss_reg_sum = 0.0; reg_cnt = 0
        for xb, yb_cls, yb_reg, _ in loader:
            xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
            logit, pred_reg = model(xb)
            if task in ("cls","multi"):
                mask_cls = yb_cls >= 0
                if mask_cls.any():
                    pred = logit[mask_cls].argmax(dim=1)
                    correct += (pred == yb_cls[mask_cls]).sum().item()
                    total += mask_cls.sum().item()
            if task in ("reg","multi"):
                mask_reg = torch.isfinite(yb_reg)
                if mask_reg.any():
                    loss_reg_sum += torch.abs(pred_reg[mask_reg] - yb_reg[mask_reg]).sum().item()
                    reg_cnt += mask_reg.sum().item()
    if task in ("cls","multi"):
        out["acc"] = correct / total if total>0 else float("nan")
    if task in ("reg","multi"):
        out["mae"] = loss_reg_sum / reg_cnt if reg_cnt>0 else float("nan")
    return out

# ----------------------
# Logger
# ----------------------
def setup_logger(args):
    # 產生唯一 log 資料夾與檔名
    now = datetime.now(timezone(timedelta(hours=8)))  # 台北時間
    run_name = f"train_{now.strftime('%Y%m%d_%H%M%S')}_{args.symbols.replace(',','_')}"
    run_dir = Path("logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "dump.log"

    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("=== LSTM Training Start ===")
    logging.info(f"台北時間: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(
        f"參數: symbols={args.symbols}, start={args.start}, end={args.end}, "
        f"interval={args.interval}min, seq_len={args.seq_len} bars, "
        f"batch_size={args.batch_size}, task={args.task}, data_mode={args.data_mode}, "
        f"layers={args.layers}, hidden={args.hidden}, dropout={args.dropout}, "
        f"epochs={args.epochs}, patience={args.patience}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, val_ratio={args.val_ratio}, "
        f"test_ratio={args.test_ratio}, ckpt={run_dir / 'lstm.ckpt'}"
    )
    return run_dir

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, default="features")
    ap.add_argument("--labels_root", type=str, default="labels")
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--symbols", type=str, default="BTCUSDT")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)

    ap.add_argument("--task", type=str, default="auto", choices=["auto","cls","reg","multi"])
    ap.add_argument("--label_cls", type=str, default="y_cls_sign_60m")
    ap.add_argument("--label_reg", type=str, default="y_reg_ret_60m")
    ap.add_argument("--alpha", type=float, default=0.5)

    # 修改預設值，360分鐘(6小時) -> 36根10分鐘K線
    ap.add_argument("--seq_len", type=int, default=36,
                   help="序列長度 (以重採樣後的K線根數計算)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--ckpt", type=str, default="lstm.ckpt")
    ap.add_argument("--data_mode", type=str, default="eager", choices=["eager", "lazy"],
                   help="資料產生方式: eager=一次展開全部, lazy=動態產生序列(省記憶體)")
    # 增加重採樣資訊參數 (optional，用於記錄)
    ap.add_argument("--interval", type=int, default=10,
                   help="資料重採樣區間 (分鐘)")
    ap.add_argument("--weight_mode", type=str, default="none", choices=["none", "time"],
                   help="Loss weighting mode")
    ap.add_argument("--half_life_days", type=float, default=90.0,
                   help="Half-life in days for time-based weighting")
    
    args = ap.parse_args()

    run_dir = setup_logger(args)
    ckpt_path = run_dir / "lstm.ckpt"

    features_root = Path(args.features_root)
    labels_root = Path(args.labels_root)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Load & merge
    frames = []
    for sym in symbols:
        print(f"Processing {sym}...")
        df_all = load_merged_months(
            features_root=Path(args.features_root), 
            labels_root=Path(args.labels_root),
            exchange=args.exchange,
            symbol=sym,
            start=args.start,
            end=args.end,
            interval=args.interval
        )
        if df_all.empty:
            print(f"[warn] no merged rows for {sym}")
            continue
        frames.append(df_all)
    if not frames:
        raise SystemExit("No data found.")

    df_all = pd.concat(frames, ignore_index=True).sort_values(["symbol","open_time"]).reset_index(drop=True)
    # print("All columns in df_all:")
    # print(df_all.columns.tolist())

    # 獲取特徵分類和完整特徵列表
    features_by_category, feats = infer_feature_columns(df_all, exclude=[args.label_cls, args.label_reg])
    
    # 輸出詳細的特徵統計信息
    logging.info(f"=== 特徵統計信息 ===")
    logging.info(f"總特徵數量: {len(feats)}")
    logging.info(f"\n完整特徵列表: {feats}")  # 添加這行來看具體的特徵列表
    for category, features in features_by_category.items():
        if features:  # 只輸出非空類別
            logging.info(f"\n{category.upper()} 特徵 ({len(features)} 個):")
            for feat in sorted(features):
                logging.info(f"  - {feat}")
    
    logging.info(f"\n=== 標籤信息 ===")
    if args.label_cls in df_all.columns:
        unique_cls = df_all[args.label_cls].dropna().unique()
        logging.info(f"分類標籤 ({args.label_cls}): {len(unique_cls)} 個不同類別")
        logging.info(f"  - 類別值: {sorted(unique_cls)}")
    if args.label_reg in df_all.columns:
        reg_stats = df_all[args.label_reg].describe()
        logging.info(f"回歸標籤 ({args.label_reg}):")
        logging.info(f"  - 均值: {reg_stats['mean']:.4f}")
        logging.info(f"  - 標準差: {reg_stats['std']:.4f}")
        logging.info(f"  - 範圍: [{reg_stats['min']:.4f}, {reg_stats['max']:.4f}]")

    if args.weight_mode == "time":
        logging.info(f"[info] time-weighting enabled (half_life = {args.half_life_days:.1f} days)")

    if args.data_mode == "eager":
        # Build sequences per symbol (原本做法)
        Xs, Ys_c, Ys_r = [], [], []
        # df_all.to_csv("debug_merged.csv", index=False)
        for sym, dfg in df_all.groupby("symbol"):
            X, yc, yr = make_sequences(dfg, feats, {"cls": args.label_cls, "reg": args.label_reg}, args.seq_len)
            if len(X):
                Xs.append(X); Ys_c.append(yc); Ys_r.append(yr)
        X = np.concatenate(Xs, axis=0)
        y_cls = np.concatenate(Ys_c, axis=0) if Ys_c else np.array([])
        y_reg = np.concatenate(Ys_r, axis=0) if Ys_r else np.array([])
        print(f"[task] X: {X.shape}")
        print(f"[task] y_cls: {y_cls.shape}")
        print(f"[task] y_reg: {y_reg.shape}")

        # time split
        n = len(X)
        n_train = int(n * (1 - args.val_ratio - args.test_ratio))
        n_val = int(n * args.val_ratio)
        sl_tr = slice(0, n_train); sl_va = slice(n_train, n_train+n_val); sl_te = slice(n_train+n_val, None)
        X_tr, X_va, X_te = X[sl_tr], X[sl_va], X[sl_te]
        yc_tr, yc_va, yc_te = y_cls[sl_tr], y_cls[sl_va], y_cls[sl_te]
        yr_tr, yr_va, yr_te = y_reg[sl_tr], y_reg[sl_va], y_reg[sl_te]

        # standardize
        mu = X_tr.mean(axis=(0,1), keepdims=True)
        st = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / st; X_va = (X_va - mu) / st; X_te = (X_te - mu) / st

        ds_tr = SeqDataset(X_tr, yc_tr, yr_tr, timestamps=df_all["open_time"].values[sl_tr], weight_mode=args.weight_mode, half_life_days=args.half_life_days, task=args.task)
        ds_va = SeqDataset(X_va, yc_va, yr_va, task=args.task)
        ds_te = SeqDataset(X_te, yc_te, yr_te, task=args.task)
        task = ds_tr.task
        n_classes = len(ds_tr.cls_label_map) if ds_tr.cls_label_map is not None else 0
        if task in ("cls","multi") and n_classes == 0:
            uniq = sorted(np.unique(yc_tr[~pd.isna(yc_tr)])); n_classes = len(uniq) if len(uniq)>0 else 3
        print(f"[task] {task} | classes={n_classes if task!='reg' else 'N/A'}")

    else:
        # Lazy 模式，直接用 LazySeqDataset
        print("[info] 使用 LazySeqDataset (動態產生序列，省記憶體)")
        # 先依時間排序
        df_all = df_all.sort_values(["symbol", "open_time"]).reset_index(drop=True)
        n = len(df_all)
        n_train = int(n * (1 - args.val_ratio - args.test_ratio))
        n_val = int(n * args.val_ratio)
        sl_tr = slice(0, n_train); sl_va = slice(n_train, n_train+n_val); sl_te = slice(n_train+n_val, None)
        df_tr = df_all.iloc[sl_tr].copy()
        df_va = df_all.iloc[sl_va].copy()
        df_te = df_all.iloc[sl_te].copy()

        # 標準化統計量只用訓練集計算
        arr_tr = df_tr[feats].to_numpy(dtype=np.float32)
        mu = arr_tr.mean(axis=0, keepdims=True)
        st = arr_tr.std(axis=0, keepdims=True) + 1e-8

        # 標準化
        for df_ in [df_tr, df_va, df_te]:
            df_[feats] = df_[feats].astype(np.float32)
            df_.loc[:, feats] = (df_[feats] - mu) / st

        ds_tr = LazySeqDataset(df_tr, feats, {"cls": args.label_cls, "reg": args.label_reg}, args.seq_len, weight_mode=args.weight_mode, half_life_days=args.half_life_days, task=args.task)
        ds_va = LazySeqDataset(df_va, feats, {"cls": args.label_cls, "reg": args.label_reg}, args.seq_len, task=args.task)
        ds_te = LazySeqDataset(df_te, feats, {"cls": args.label_cls, "reg": args.label_reg}, args.seq_len, task=args.task)
        task = ds_tr.task
        n_classes = len(ds_tr.cls_label_map) if ds_tr.cls_label_map is not None else 0
        if task in ("cls","multi") and n_classes == 0:
            uniq = sorted(df_tr[args.label_cls].dropna().unique()); n_classes = len(uniq) if len(uniq)>0 else 3
        print(f"[task] {task} | classes={n_classes if task!='reg' else 'N/A'}")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task] device: {device}")
    backbone = LSTMBackbone(input_dim=len(feats), hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    model = LSTMHeads(backbone, n_classes=max(n_classes,1)).to(device)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # print(f"[data] train: {len(ds_tr)} | val: {len(ds_va)} | test: {len(ds_te)} | batch_size: {args.batch_size}")

    # xb, yb_cls, yb_reg = next(iter(train_loader))
    # print(xb.shape)
    
    model = train_model(
        model, (train_loader, val_loader), device, task=task, epochs=args.epochs,
        patience=args.patience, lr=args.lr, weight_decay=args.weight_decay,
        alpha=args.alpha, ckpt=str(ckpt_path)
    )

    # evaluation
    metrics = evaluate(model, test_loader, device, task)
    print(f"[test] {metrics}")

    # artefacts
    np.save(run_dir / "feature_mean.npy", mu.squeeze(0))
    np.save(run_dir / "feature_std.npy", st.squeeze(0))
    with open(run_dir / "feature_list.txt", "w", encoding="utf-8") as f:
        for c in feats: f.write(str(c) + "\n")
    if hasattr(ds_tr, "cls_label_map") and ds_tr.cls_label_map is not None:
        with open(run_dir / "class_map.json", "w", encoding="utf-8") as f:
            json.dump(ds_tr.cls_label_map, f, ensure_ascii=False, indent=2)
    print(f"[done] saved artefacts to {run_dir}")

    logging.info(f"訓練完成，metrics: {metrics}")
    logging.info("=== LSTM Training End ===")
    logging.info(f"Log dir: {run_dir}")
    
if __name__ == "__main__":
    main()




# python train_lstm.py --features_root features --labels_root labels --symbols BTCUSDT,ETHUSDT,BNBUSDT,DOGEUSDT,SOLUSDT --start 2021-01-01 --end 2025-08-23 --label_cls y_cls_sign_60m --task cls --seq_len 36 --layers 5 --hidden 384 --dropout 0.3 --epochs 50 --patience 25 --lr 1e-3 --weight_decay 1e-4 --batch_size 512 --data_mode lazy --interval 10 --weight_mode time --half_life_days 90