#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_transform.py  (Transformer-ready preprocessing + feature engineering)
---------------------------------------------------------------------------

【核心設計原則】
1. 時序安全的去冗餘 + 結構化 encoding + 嚴格切割
2. Feature 與 Label 完全解耦（防止未來信息洩漏）
3. Transformer 專用 normalization（rolling z-score）
4. Symbol embedding 在模型層完成，preprocessing 不處理

【Feature 分層結構】
   ├─ A. Meta（不進模型）
   │  ├─ symbol, open_time, exchange, year, month
   │
   ├─ B. 原始尺度特徵（需 ratio/log 化）
   │  ├─ close, volume, number_of_trades, obv
   │
   ├─ C. 報酬類（保留 fat tail）
   │  ├─ ret_1m, ret_3m, ret_6m, ret_12m
   │  ├─ ret_1m_btc_ctx, vol_roc_*
   │
   ├─ D. 趨勢狀態（高度共線，需要選擇性保留）
   │  ├─ close_norm_20, close_norm_50 (不用全部 SMA)
   │  ├─ ema_*_slope, ema_*_dev
   │  ├─ bb_width, bb_pos, don_width, don_break_dist
   │
   ├─ E. 波動/風險（非常有價值，適度去冗餘）
   │  ├─ atr_14, rv_60, rv_240, rv_60_pctile, park_60
   │
   ├─ F. 量能/資金流（完整保留）
   │  ├─ vol_z_30, vol_z_60, trades_z_60
   │  ├─ obv_slope_60, cmf_20, mfi_14, close_vwap_dev
   │
   ├─ G. 週期/時間（已 encoded）
   │  ├─ hour_sin, hour_cos, dow_*
   │
   └─ H. Labels（絕禁進 feature pipeline）
      ├─ y_reg_ret_*, y_cls_sign_*
      ├─ y_tp_sl_*, y_tb_*
      └─ *_t_hit（僅分析用）

【Normalization 策略】
   • 報酬類特徵：保持原樣（fat tail）
   • 狀態特徵：rolling z-score（過去 500 bars）
   • 量能特徵：rolling z-score（過去 500 bars）
   • 波動特徵：rolling z-score（過去 500 bars）
   • 時間特徵：不標準化

【Pipeline 流程】
   load_features() → load_labels() → align_on_open_time() 
   → drop_labels_from_features() → feature_reduce() → rolling_normalize()
   → build_sequences() → time_split() → save/return
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
from tool.data_loader import load_data


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
    ignore = {"symbol", "open_time", "year", "month", "exchange"} | set(exclude)
    
    # 排除標籤相關列
    label_prefixes = ["y_cls_", "y_reg_", "y_tp_sl_", "y_tb_", "y_vol_", "_t_hit"]
    
    # 更新需要排除的列
    ignore.update([col for col in df.columns if any(col.startswith(prefix) for prefix in label_prefixes)])
    
    # 定義各類特徵的前綴或關鍵字 (依照 feature_builder.py 的實際輸出)
    feature_patterns = {
        "base": ["close", "volume", "number_of_trades"],
        "returns": ["ret_1m"],
        "technical": ["sma_", "rsi_14", "atr_14", "atr_30", "atr_60"],
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
    logging.info("\n=== Feature Selection Debug ===")
    logging.info(f"All columns: {df.columns.tolist()}")
    logging.info("\nFeatures by category:")
    for cat, feats in features_by_category.items():
        logging.info(f"{cat}: {feats}")
    logging.info(f"\nTotal features selected: {len(all_features)}")
    logging.info(f"Selected features: {all_features}")
    logging.info("===========================\n")

    return features_by_category, all_features


# ========================================================================================
# 【新增】結構化 Feature Reduction（基於設計規格的去冗餘）
# ========================================================================================

def reduce_features_transformer(features_by_category: dict) -> list:
    """
    根據 Transformer 的需求對特徵進行選擇性去冗餘。
    
    核心原則：
    - 同一「經濟意義」的特徵，只保留「最有表達力的 1~2 個」
    - 報酬類特徵：完整保留
    - 趨勢狀態：只保留位置、斜率、寬度（不保留所有 SMA）
    - 波動特徵：選擇不同時間窗口的代表特徵
    - 量能特徵：完整保留
    
    Args:
        features_by_category: 按類別分組的特徵字典
    
    Returns:
        list: 減少後的特徵列表
    """
    
    # A. 基礎特徵 - 需要 ratio/log 化的，暫時保留（會在 normalize 階段處理）
    base_feats = features_by_category.get("base", [])
    
    # B. 報酬類 - 完全保留（已是 scale-free）
    ret_feats = features_by_category.get("returns", [])
    
    # C. 技術指標 - 高度去冗餘
    #    原則：SMA 全砍；RSI 保留
    tech_feats = features_by_category.get("technical", [])
    tech_reduced = [f for f in tech_feats if not f.startswith("sma_")]
    
    # D. 波動特徵 - 適度去冗餘
    #    原則：保留 rv_60, rv_240, atr_14, atr_30, atr_60；砍掉短期（5,15,30）
    vol_feats = features_by_category.get("volatility", [])
    vol_keep_patterns = ["rv_60", "rv_240", "rv_60_pctile", "park_60", "rs_60", "atr_14", "atr_30", "atr_60"]
    vol_reduced = [f for f in vol_feats if any(f.startswith(p) for p in vol_keep_patterns)]
    
    # E. 量能 - 完整保留
    vol_analysis_feats = features_by_category.get("volume_analysis", [])
    
    # F. 其他 - 需要手動檢查
    other_feats = features_by_category.get("other", [])
    
    # 對其他特徵進行進一步的篩選
    other_reduced = []
    for f in other_feats:
        # 保留的關鍵詞
        keep_keywords = [
            "close_norm_",       # 價格位置
            "ema_",              # EMA 相關
            "bb_",               # Bollinger 相關
            "don_",              # Donchian 相關
            "obv",               # OBV
            "cmf_",              # Cash Flow
            "mfi_",              # Money Flow Index
            "vwap",              # VWAP
            "hour_",             # 時間特徵
            "dow_",              # 星期特徵
            "vol_roc_",          # 量能 ROC
            "_btc_ctx",          # BTC 上下文特徵
        ]
        
        if any(f.startswith(kw) or kw in f for kw in keep_keywords):
            # 進一步篩選：不保留所有 SMA、不保留短期 EMA
            if f.startswith("sma_"):
                continue  # 已由 technical 篩選掉
            if f.startswith("ema_") and not ("slope" in f or "dev" in f):
                continue  # 只保留 EMA 斜率和偏離，不保留絕對值
            other_reduced.append(f)
    
    # 組合結果
    selected = base_feats + ret_feats + tech_reduced + vol_reduced + vol_analysis_feats + other_reduced
    
    logging.info("\n=== Feature Reduction for Transformer ===")
    logging.info(f"Before: {sum(len(v) for v in features_by_category.values())} features")
    logging.info(f"After:  {len(selected)} features")
    logging.info(f"Removed: {sum(len(v) for v in features_by_category.values()) - len(selected)} redundant features")
    logging.info(f"\nSelected features: {sorted(selected)}")
    logging.info("="*50 + "\n")
    
    return selected


# ========================================================================================
# 【新增】Rolling Normalization（Transformer 專用）
# ========================================================================================

def rolling_z_score(series: pd.Series, window: int = 500, min_periods: int = 50, clip: float = 5.0) -> pd.Series:
    """
    計算 rolling z-score（僅基於過去資料，防止未來洩漏）。
    
    Args:
        series: 輸入時間序列
        window: rolling window 大小
        min_periods: 最少需要的非NaN值
        clip: 裁剪範圍（防止極端值）
    
    Returns:
        pd.Series: 正規化後的序列
    """
    mu = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mu) / (std + 1e-8)  # 加小值避免除零
    z = z.clip(-clip, clip)
    z = z.fillna(0) # 用 0 填充避免 NAN
    return z


def normalize_features_transformer(df: pd.DataFrame, feature_list: list, 
                                   window: int = 500) -> pd.DataFrame:
    """
    按分組對特徵進行 Transformer 專用的正規化。
    
    規則：
    - 報酬類特徵（ret_*）：保持原樣，不正規化（保留 fat tail）
    - 狀態類特徵（price position, slope, width）：rolling z-score
    - 波動特徵（rv_, atr_）：rolling z-score
    - 量能特徵（vol_, obv_, cmf_）：rolling z-score
    - 時間特徵（hour_, dow_）：不正規化
    
    Args:
        df: 輸入 DataFrame
        feature_list: 要正規化的特徵列表
        window: rolling window 大小
    
    Returns:
        pd.DataFrame: 正規化後的 DataFrame（in-place 修改）
    """
    df["close"] = np.log1p(df["close"].astype(float)) # 處裡 "close" 欄位存在的極大值，需要在推論時做 inverse（expm1）
    df["volume"] = np.log1p(df["volume"].astype(float)) # 處裡 "volume" 欄位存在的極大值，需要在推論時做 inverse（expm1）
    df["number_of_trades"] = np.log1p(df["number_of_trades"].astype(float)) # 處裡 "number_of_trades" 欄位存在的極大值，需要在推論時做 inverse（expm1）
    
    df_norm = df.copy()
    
    # 定義各類特徵的正規化策略
    no_normalize_keywords = ["ret_", "hour_", "dow_", "sin", "cos"]
    rolling_z_keywords = ["close_norm", "ema_", "bb_", "don_", "atr_", 
                          "rv_", "park_", "rs_", "vol_z_", "trades_z_", 
                          "obv", "cmf_", "mfi_", "vwap", "vol_roc_"]
    
    for feat in feature_list:
        if feat not in df_norm.columns:
            continue
        
        # 檢查是否需要正規化
        skip = any(kw in feat for kw in no_normalize_keywords)
        if skip:
            continue
        
        # 檢查是否屬於 rolling z-score 類
        apply_rolling_z = any(kw in feat for kw in rolling_z_keywords)
        if apply_rolling_z:
            df_norm[feat] = rolling_z_score(df[feat], window=window)
            # print(f"NaN count in df_norm[{feat}]: {df_norm[feat].isna().sum().sum()}")
    
    return df_norm


# ========================================================================================
# 【新增】Label 與 Feature 完全解耦
# ========================================================================================

def drop_labels_from_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除所有 Label 欄位，確保 feature pipeline 中不含任何未來信息。
    
    移除的欄位模式：
    - y_cls_*, y_reg_*, y_tp_sl_*, y_tb_*, y_vol_*
    - *_t_hit（時間戳記）
    
    Args:
        df: 輸入 DataFrame
    
    Returns:
        pd.DataFrame: 移除 Label 後的 DataFrame
    """
    label_patterns = ["y_cls_", "y_reg_", "y_tp_sl_", "y_tb_", "y_vol_", "_t_hit"]
    cols_to_drop = [c for c in df.columns if any(c.startswith(p) or p in c for p in label_patterns)]
    
    if cols_to_drop:
        logging.info(f"[info] Dropping {len(cols_to_drop)} label columns: {cols_to_drop}")
    
    return df.drop(columns=cols_to_drop, errors='ignore')

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
        # f"batch_size={args.batch_size}, task={args.task}, data_mode={args.data_mode}, "
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
    ap.add_argument("--label_cls", type=str, default="y_cls_sign_120m")
    ap.add_argument("--label_reg", type=str, default="y_reg_ret_120m")
    ap.add_argument("--label_vol", type=str, default=None) # "y_vol_120m"
    ap.add_argument("--label_tp", type=str, default=None) # "y_tp_sl_120m"
    ap.add_argument("--alpha", type=float, default=0.5)

    # 修改預設值，360分鐘(6小時) -> 36根10分鐘K線
    ap.add_argument("--seq_len", type=int, default=36,
                   help="序列長度 (以重採樣後的K線根數計算)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--ckpt", type=str, default="lstm.ckpt")
    # ap.add_argument("--data_mode", type=str, default="eager", choices=["eager", "lazy"],
    #                help="資料產生方式: eager=一次展開全部, lazy=動態產生序列(省記憶體)")
    # 增加重採樣資訊參數 (optional，用於記錄)
    ap.add_argument("--interval", type=int, default=10,
                   help="資料重採樣區間 (分鐘)")
    ap.add_argument("--weight_mode", type=str, default="none", choices=["none", "time"],
                   help="Loss weighting mode")
    ap.add_argument("--half_life_days", type=float, default=90.0,
                   help="Half-life in days for time-based weighting")
    
    args = ap.parse_args()

    run_dir = setup_logger(args)
    # Import transformer after logger setup so its logging goes to run_dir/dump.log
    from tool.transformer import TimeSeriesTransformerModel, train_transformer, evaluate_transformer
    ckpt_path = run_dir / "lstm.ckpt"

    features_root = Path(args.features_root)
    labels_root = Path(args.labels_root)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Load & merge
    frames = []
    for sym in symbols:
        logging.info(f"Processing {sym}...")
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
            logging.info(f"[warn] no merged rows for {sym}")
            continue
        frames.append(df_all)
    if not frames:
        raise SystemExit("No data found.")

    df_all = pd.concat(frames, ignore_index=True).sort_values(["symbol","open_time"]).reset_index(drop=True)
    logging.info("All columns in df_all:")
    logging.info(df_all.columns.tolist())
    diff = df_all["open_time"].diff().dropna()
    print(diff.value_counts())

    # ============================================================================
    # 【新】Preprocessing Pipeline for Transformer
    # ============================================================================
    
    logging.info("\n[step 1] Feature Inference & Categorization")
    features_by_category, feats_all = infer_feature_columns(df_all, exclude=[args.label_cls, args.label_reg])
    
    logging.info("\n[step 2] Feature Reduction (去冗餘)")
    feats_reduced = reduce_features_transformer(features_by_category)
    
    logging.info("\n[step 3] Drop Labels from Features (feature ↔ label 完全解耦)")
    df_features_only = drop_labels_from_features(df_all)
    
    logging.info("\n[step 4] Rolling Normalization (Transformer 專用正規化)")
    df_normalized = normalize_features_transformer(df_features_only, feats_reduced, window=500)

    # print(f"NaN count in df_normalized: {df_normalized.isna().sum().sum()}")

    # ============================================================================
    # 【新】提取分開的 Label 數據（永遠保持獨立）
    # ============================================================================
    
    # Meta 列（對齊用）
    meta_cols = ["symbol", "open_time"]
    
    # Label 列
    label_cols = [c for c in df_all.columns if c.startswith("y_")]
    
    # 建立 feature 和 label 的分離視圖
    df_X = df_normalized[[c for c in df_normalized.columns if c in meta_cols or c in feats_reduced]].copy()
    df_y = df_all[[c for c in df_all.columns if c in meta_cols or c in label_cols]].copy()
    
    logging.info("\n[step 5] Final Feature Statistics")
    logging.info(f"✓ Feature matrix shape: {df_X.shape}")
    logging.info(f"✓ Feature columns: {len(feats_reduced)}")
    logging.info(f"  - Columns: {feats_reduced}")
    logging.info(f"✓ Label matrix shape: {df_y.shape}")
    logging.info(f"✓ Label columns: {len(label_cols)}")
    logging.info(f"  - Columns: {label_cols}")
    
    # ============================================================================
    # 【新】詳細日誌輸出
    # ============================================================================
    
    logging.info("\n[step 6] Feature Categorization Summary (After Reduction)")
    logging.info(f"{'='*70}")
    
    # 重新分類（基於減少後的特徵）
    feats_reduced_set = set(feats_reduced)
    summary = {
        "returns": [f for f in feats_reduced if f.startswith("ret_")],
        "trend_state": [f for f in feats_reduced if any(x in f for x in ["close_norm", "ema_", "bb_", "don_"])],
        "volatility": [f for f in feats_reduced if any(x in f for x in ["rv_", "atr_", "park_", "rs_"])],
        "volume": [f for f in feats_reduced if any(x in f for x in ["vol_", "trades_z_", "obv", "cmf_", "mfi_"])],
        "time_cycle": [f for f in feats_reduced if any(x in f for x in ["hour_", "dow_"])],
        "context": [f for f in feats_reduced if "btc_ctx" in f or "roc_" in f],
    }
    
    for cat, feats_in_cat in summary.items():
        if feats_in_cat:
            logging.info(f"\n{cat.upper():20} ({len(feats_in_cat)} 個)")
            for f in sorted(feats_in_cat):
                logging.info(f"  ✓ {f}")

    logging.info(f"\n{'='*70}")
    logging.info(f"\n【預處理完成】")
    logging.info(f"  • 原始特徵: {len(feats_all)} → 減少後: {len(feats_reduced)}")
    logging.info(f"  • 移除冗餘: {len(feats_all) - len(feats_reduced)} 個特徵")
    logging.info(f"  • Label 行數: {len(df_y)}")
    logging.info(f"  • 時間範圍: {df_X['open_time'].min()} → {df_X['open_time'].max()}")
    
    if args.weight_mode == "time":
        logging.info(f"  • 時間加權: 開啟 (半衰期 = {args.half_life_days:.1f} 天)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_reduced = pd.merge(df_X, df_y, on=["symbol","open_time"], how="inner")

    # determine optional label names for tp / vol if present in df_y
    label_tp = args.label_tp
    label_vol = args.label_vol

    train_loader, val_loader, test_loader, n_classes, n_tp_classes = load_data(
        df_all=df_reduced,
        feats=feats_reduced,
        label_cls=args.label_cls,
        label_reg=args.label_reg,
        label_tp=label_tp,
        label_vol=label_vol,
        weight_mode=args.weight_mode,
        half_life_days=args.half_life_days,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    model = TimeSeriesTransformerModel(
        input_dim=len(feats_reduced),
        d_model=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.layers,
        seq_len=args.seq_len,
        n_cls=n_classes,
        n_tp=n_tp_classes,
        n_symbols=len(symbols),
        symbol_embedding_dim=32,
        use_symbol_emb=True,
        use_intermediate_heads=True,
        use_cls=(args.label_cls is not None),
        use_reg=(args.label_reg is not None),
        use_vol=(label_vol is not None),
        use_tp=(label_tp is not None)
    ).to(device)

    task_heads = []
    if args.label_cls is not None:
        task_heads.append("cls")
    if args.label_reg is not None:
        task_heads.append("reg")
    if label_vol is not None:
        task_heads.append("vol")
    if label_tp is not None:
        task_heads.append("tp")

    
    # max_batches = 10  # 要檢查多少個 batch，改成你要的數字
    # for bi, batch in enumerate(train_loader):
    #     xb = batch[0]            # (B, seq_len, input_dim)
    #     y_cls = batch[1]
    #     y_reg = batch[2]
    #     y_vol = batch[3]
    #     y_tp = batch[4]
    #     w = batch[5] if len(batch) > 5 else torch.ones_like(y_cls, dtype=torch.float32)
    #     xb_np = xb.numpy()
    #     B, S, D = xb_np.shape
    #     print(f"batch {bi}: xb shape={xb_np.shape}")
    #     print("  global: min,max,mean,std =", xb_np.min(), xb_np.max(), xb_np.mean(), xb_np.std())
    #     # 每個 feature（沿時間聚合）的統計：max/min/mean, count NaN, count extreme
    #     feat_max = xb_np.max(axis=1).max(axis=0)    # over seq then batch -> per-feature max
    #     feat_min = xb_np.min(axis=1).min(axis=0)
    #     feat_mean = xb_np.mean(axis=(0,1))
    #     feat_nan_counts = np.isnan(xb_np).sum(axis=(0,1))
    #     extreme_counts = (np.abs(xb_np) > 1e6).sum(axis=(0,1))
    #     # 顯示有 NaN 或 極端值 的 feature indices
    #     bad_nan_idx = np.where(feat_nan_counts > 0)[0]
    #     bad_ext_idx = np.where(extreme_counts > 0)[0]
    #     print(bad_ext_idx)
    #     if bad_nan_idx.size:
    #         print("  features with NaNs:", bad_nan_idx, "counts:", feat_nan_counts[bad_nan_idx])
    #     if bad_ext_idx.size:
    #         print("  features with >1e6 values:", bad_ext_idx, "counts:", extreme_counts[bad_ext_idx])
    #     # 若只想看前幾個 feature 的數值
    #     for f in range(min(5, D)):
    #         print(f"   feat[{f}]: min={feat_min[f]:.3e} max={feat_max[f]:.3e} mean={feat_mean if isinstance(feat_mean, float) else feat_mean[f]:.3e} nan={feat_nan_counts[f]} extreme={extreme_counts[f]}")
    #     if bi + 1 >= max_batches:
    #         break

    model = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        task_heads=task_heads,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_weights={"cls": 1.0, "reg": 0.5, "vol": 0.2, "tp": 0.5},
        ckpt=ckpt_path
    )

    # run evaluation on test set
    metrics = evaluate_transformer(model, test_loader, device, task_heads)
    logging.info(f"[test] {metrics}")
    logging.info(f"[test] {metrics}")

    # ----------------------
    # Artefacts (feature list, class map, metadata)
    # ----------------------
    # feature list
    with open(run_dir / "feature_list.txt", "w", encoding="utf-8") as f:
        for c in feats_reduced:
            f.write(str(c) + "\n")

    # class maps (cls / tp) from dataset if available
    class_maps = {}
    try:
        ds_train = train_loader.dataset
        if hasattr(ds_train, "cls_label_map") and ds_train.cls_label_map is not None:
            class_maps["cls"] = ds_train.cls_label_map
        if hasattr(ds_train, "tp_label_map") and ds_train.tp_label_map is not None:
            class_maps["tp"] = ds_train.tp_label_map
    except Exception:
        # best-effort only
        pass

    if class_maps:
        # Ensure all keys/values are JSON-serializable (convert numpy types, int64 keys, etc.)
        def _make_json_serializable(o):
            if isinstance(o, dict):
                new = {}
                for k, v in o.items():
                    # normalize key to a JSON-friendly type
                    if isinstance(k, (np.integer,)):
                        k2 = int(k)
                    elif not isinstance(k, (str, int, float, bool, type(None))):
                        k2 = str(k)
                    else:
                        k2 = k
                    new[k2] = _make_json_serializable(v)
                return new
            if isinstance(o, (list, tuple)):
                return [_make_json_serializable(x) for x in o]
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        safe_maps = _make_json_serializable(class_maps)
        with open(run_dir / "class_map.json", "w", encoding="utf-8") as f:
            json.dump(safe_maps, f, ensure_ascii=False, indent=2)

    # metadata
    meta = {
        "seq_len": args.seq_len,
        "task_heads": task_heads,
        "n_classes": n_classes,
        "n_tp_classes": n_tp_classes,
        "n_features": len(feats_reduced),
        "norm": "rolling_z_score"
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"[done] saved artefacts to {run_dir}")
    logging.info(f"訓練完成，metrics: {metrics}")
    logging.info("=== Transformer Training End ===")
    logging.info(f"Log dir: {run_dir}")

if __name__ == "__main__":
    main()




# python train_transformer.py --features_root features --labels_root labels --symbols BTCUSDT,ETHUSDT,BNBUSDT,DOGEUSDT,SOLUSDT --start 2021-01-01 --end 2025-08-23 --label_cls y_cls_sign_60m --task cls --seq_len 36 --layers 5 --hidden 256 --n_heads 8 --dropout 0.3 --epochs 50 --patience 25 --lr 1e-3 --weight_decay 1e-4 --batch_size 512 --interval 10 --weight_mode none
# python train_transformer.py --features_root features --labels_root labels --symbols BTCUSDT,ETHUSDT,BNBUSDT,DOGEUSDT,SOLUSDT --start 2021-01-01 --end 2025-08-23 --seq_len 36 --layers 5 --hidden 256 --n_heads 8 --dropout 0.3 --epochs 50 --patience 25 --lr 1e-3 --weight_decay 1e-4 --batch_size 512 --interval 30 --weight_mode none
