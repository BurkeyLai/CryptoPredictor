import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class LazySeqDataset(Dataset):
    """
    動態產生序列，避免一次將所有序列展開到記憶體。
    """
    def __init__(self, df, feats, targets, seq_len, weight_mode="none", half_life_days=90.0):
        self.df = df
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

            # 標籤轉 numpy（新增 vol / tp 支援）
            label_dict = {}
            if targets.get("cls") in dfg.columns:
                label_dict["cls"] = dfg[targets["cls"]].to_numpy()
            if targets.get("reg") in dfg.columns:
                label_dict["reg"] = dfg[targets["reg"]].to_numpy()
            if targets.get("vol") in dfg.columns:
                # 波動率類（連續值）
                label_dict["vol"] = dfg[targets["vol"]].to_numpy()
            if targets.get("tp") in dfg.columns:
                # take-profit / stop-loss 類（分類）
                label_dict["tp"] = dfg[targets["tp"]].to_numpy()
            self.symbol_labels[sym] = label_dict
            
            # 產生可用的索引
            for i in range(len(dfg) - seq_len + 1):
                self.indices.append((sym, i))
        
        # 預先計算 label map（僅限分類）
        # 分類標籤 map（cls / tp 各自判斷）
        y_cls_all = []
        y_tp_all = []
        for sym, labels in self.symbol_labels.items():
            if "cls" in labels:
                y_cls_all.extend(labels["cls"][self.seq_len-1:])
            if "tp" in labels:
                y_tp_all.extend(labels["tp"][self.seq_len-1:])
        y_cls_all = np.array(y_cls_all)
        y_tp_all = np.array(y_tp_all)

        # 設定分類標籤對應
        self.cls_label_map = None
        if len(y_cls_all) > 0 and (
            np.issubdtype(y_cls_all.dtype, np.integer) or
            set(np.unique(y_cls_all[~pd.isna(y_cls_all)])).issubset({-1,0,1,2,3})
        ):
            uniq = sorted(np.unique(y_cls_all[~pd.isna(y_cls_all)]))
            self.cls_label_map = {v:i for i,v in enumerate(uniq)}

        self.tp_label_map = None
        if len(y_tp_all) > 0 and (
            np.issubdtype(y_tp_all.dtype, np.integer) or
            set(np.unique(y_tp_all[~pd.isna(y_tp_all)])).issubset({-1,0,1,2,3})
        ):
            uniq_tp = sorted(np.unique(y_tp_all[~pd.isna(y_tp_all)]))
            self.tp_label_map = {v:i for i,v in enumerate(uniq_tp)}
        
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

        # task parameter removed; behavior now inferred from provided label columns
        self.task = None

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
        y_vol = labels["vol"][label_idx] if "vol" in labels else np.nan
        y_tp = labels["tp"][label_idx] if "tp" in labels else np.nan
        """
        horizon_steps = 4
        if idx < 5:
            print(start_idx)
            print(label_idx)
            print(y_reg)
            print(self.symbol_labels[sym]["reg"][label_idx])

            t = start_idx + self.seq_len - 1
            future_t = t + horizon_steps

            print("seq_start:", self.df.iloc[start_idx]["open_time"])
            print("seq_end:", self.df.iloc[t]["open_time"])
            print("label_time:", self.df.iloc[future_t]["open_time"])

            label_ret = self.df.iloc[t]["y_reg_ret_120m"]
            print("stored y_reg_ret_120m:", label_ret)

            future_window = self.df.iloc[t : future_t]["close"]

            print("future_window length:", len(future_window))
            print("future_window time range:", self.df.iloc[t]["open_time"], "→", self.df.iloc[future_t-1]["open_time"])
            print("future_min_time:", self.df.iloc[future_window.idxmin()]["open_time"])

            price_now = self.df.iloc[t]["close"]
            future_max = future_window.max()
            future_min = future_window.min()

            ret1 = (future_max - price_now) / price_now * 100
            ret2 = (future_min - price_now) / price_now * 100

            manual_ret = ret1 if abs(ret1) > abs(ret2) else ret2

            print("manual_ret:", manual_ret)
            print("diff:", manual_ret - label_ret)

            tmp_max = self.df["close"].iloc[t : t + horizon_steps].max()
            tmp_min = self.df["close"].iloc[t : t + horizon_steps].min()

            tmp_ret1 = (tmp_max - price_now) / price_now * 100
            tmp_ret2 = (tmp_min - price_now) / price_now * 100
            tmp_ret = tmp_ret1 if abs(tmp_ret1) > abs(tmp_ret2) else tmp_ret2

            print("recalc_ret:", tmp_ret)

            # x = X[idx:idx+self.seq_len]
            # y = y_reg[t]
            # print(f"X shape: {x.shape} | y shape: {y.shape} | y_cls: {y_cls} | y_reg: {y_reg}")
            # print(f"x: {x}")
            # print(f"y: {y}")

            # sample_x = X[100]
            # sample_y = y[100]
            # print(f"Sample X[100]: {sample_x}")
            # print(f"Sample y[100]: {sample_y}")
        """
        # 分類 label 映射 (cls / tp)
        if self.cls_label_map is not None and not np.isnan(y_cls):
            y_cls_idx = self.cls_label_map.get(y_cls, -1)
        else:
            y_cls_idx = -1

        if self.tp_label_map is not None and not np.isnan(y_tp):
            y_tp_idx = self.tp_label_map.get(y_tp, -1)
        else:
            y_tp_idx = -1

        w = self.weights[idx] if self.weights is not None else 1.0
        return (
            torch.from_numpy(X).float(),
            torch.tensor(y_cls_idx, dtype=torch.int64),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_vol, dtype=torch.float32),
            torch.tensor(y_tp_idx, dtype=torch.int64),
            torch.tensor(w, dtype=torch.float32)
        )

def split_by_time(df, val_ratio=0.1, test_ratio=0.1):
    # Return train/val/test as DataFrames split per-symbol in time order
    df_sorted = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    train_idx_list, val_idx_list, test_idx_list = [], [], []
    train_ratio = 1.0 - val_ratio - test_ratio

    for _, g in df_sorted.groupby("symbol", sort=False):
        n = len(g)
        if n == 0:
            continue
        t_end = int(n * train_ratio)
        v_end = int(n * (train_ratio + val_ratio))

        idx = g.index.to_numpy()
        train_idx_list.append(idx[:t_end])
        val_idx_list.append(idx[t_end:v_end])
        test_idx_list.append(idx[v_end:])

    # concatenate, handle possibility of empty lists
    def _concat_or_empty(lst):
        if not lst:
            return np.array([], dtype=int)
        return np.concatenate(lst)

    train_idx = _concat_or_empty(train_idx_list)
    val_idx = _concat_or_empty(val_idx_list)
    test_idx = _concat_or_empty(test_idx_list)

    df_train = df_sorted.loc[train_idx].reset_index(drop=True)
    df_val = df_sorted.loc[val_idx].reset_index(drop=True)
    df_test = df_sorted.loc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test

def load_data(df_all,
              feats,
              label_cls,
              label_reg,
              label_tp=None,
              label_vol=None,
              weight_mode="none",
              half_life_days = 90,
              val_ratio = 0.1,
              test_ratio = 0.1,
              seq_len = 36,
              batch_size = 512):
    # Lazy 模式，直接用 LazySeqDataset
    print("[info] 使用 LazySeqDataset (動態產生序列，省記憶體)")
    # 先依時間排序
    df_tr, df_va, df_te = split_by_time(
        df_all,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    # df_tr.to_csv("tr.csv", index=False)
    # df_va.to_csv("va.csv", index=False)
    # df_te.to_csv("te.csv", index=False)

    # 標準化
    for df_ in [df_tr, df_va, df_te]:
        df_[feats] = df_[feats].astype(np.float32)

    targets_map = {}
    if label_cls is not None:
        targets_map["cls"] = label_cls
    if label_reg is not None:
        targets_map["reg"] = label_reg
    if label_tp is not None:
        targets_map["tp"] = label_tp
    if label_vol is not None:
        targets_map["vol"] = label_vol

    ds_tr = LazySeqDataset(df_tr, feats, targets_map, seq_len, weight_mode=weight_mode, half_life_days=half_life_days)
    ds_va = LazySeqDataset(df_va, feats, targets_map, seq_len)
    ds_te = LazySeqDataset(df_te, feats, targets_map, seq_len)

    # Determine number of classes from dataset label map or raw dataframe
    n_classes = len(ds_tr.cls_label_map) if ds_tr.cls_label_map is not None else 0
    if n_classes == 0 and label_cls is not None:
        uniq = sorted(df_tr[label_cls].dropna().unique())
        n_classes = len(uniq) if len(uniq) > 0 else 3

    n_tp_classes = len(ds_tr.tp_label_map) if ds_tr.tp_label_map is not None else 0
    if n_tp_classes == 0 and label_tp is not None:
        uniq_tp = sorted(df_tr[label_tp].dropna().unique())
        n_tp_classes = len(uniq_tp) if len(uniq_tp) > 0 else 3

    print(f"[data] cls_classes={n_classes} | tp_classes={n_tp_classes}")

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=4)
    # print(f"[data] train: {len(ds_tr)} | val: {len(ds_va)} | test: {len(ds_te)} | batch_size: {args.batch_size}")

    return train_loader, val_loader, test_loader, n_classes, n_tp_classes

def main():
    pass

if __name__=='__main__':
    main()