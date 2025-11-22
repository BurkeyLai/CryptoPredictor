import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--weight_path", type=str, default="lstm.ckpt")
    ap.add_argument("--input_dim", type=int, default=34)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--month", type=int, default=9)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=36)
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1️⃣ 匯入你現有的模型結構 =====
    backbone = LSTMBackbone(input_dim=args.input_dim, 
                            hidden=args.hidden, 
                            layers=args.layers, 
                            dropout=args.dropout)
    model = LSTMHeads(backbone, n_classes=3).to(device)

    # ===== 2️⃣ 載入模型權重 =====
    ckpt = torch.load(args.weight_path + "lstm.ckpt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ===== 3️⃣ 讀取特徵資料 =====
    feature_path = Path(
        f"features/exchange=binance/{args.interval}min/"
        f"symbol={args.symbol}/year={args.year}/"
        f"month={args.month:02d}/"
        f"features-{args.year}-{args.month:02d}.parquet"
    )
    df = pd.read_parquet(feature_path)
    df = df.sort_values("open_time").reset_index(drop=True)

    # ===== 4️⃣ 標準化特徵 (使用訓練時保存的 mean/std) =====
    feature_list = [l.strip() 
                   for l in open(args.weight_path + "feature_list.txt").readlines()]
    mu = np.load(args.weight_path + "feature_mean.npy")
    st = np.load(args.weight_path + "feature_std.npy")

    # 檢查特徵是否存在並建立映射
    available_features = []
    feature_indices = []  # 新增: 記錄特徵在原始列表中的索引
    for i, feat in enumerate(feature_list):
        if feat in df.columns:
            available_features.append(feat)
            feature_indices.append(i)
        else:
            print(f"[warning] Feature '{feat}' not found in data")

    if len(available_features) != len(feature_list):
        print(f"[warning] Using {len(available_features)}/{len(feature_list)} features")
        print("[warning] Available features:", available_features)
    
    if not available_features:
        raise ValueError("No valid features found in data!")

    # 只使用存在的特徵對應的 mu 和 st
    feature_indices = np.array(feature_indices)
    mu_subset = mu[feature_indices]
    st_subset = st[feature_indices]
    
    # 確保維度正確
    X_all = df[available_features].to_numpy(dtype=np.float32)
    print(f"[info] X_all shape: {X_all.shape}, mu shape: {mu_subset.shape}, st shape: {st_subset.shape}")
    
    # 標準化
    X_all = (X_all - mu_subset) / st_subset

    # ===== 5️⃣ 構造序列 (seq_len=36) =====
    try:
        seqs = np.lib.stride_tricks.sliding_window_view(
            X_all, (args.seq_len, X_all.shape[1]))[:, 0, :, :]
        print(f"[info] sequences shape: {seqs.shape}")
    except Exception as e:
        raise ValueError(f"Error creating sequences: {str(e)}\nData shape: {X_all.shape}")

    # ===== 6️⃣ 模型推論 =====
    probs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            xb = torch.from_numpy(seqs[i:i+args.batch_size]).to(device)
            logit, _ = model(xb)
            p = torch.softmax(logit, dim=1).cpu().numpy()
            probs.append(p)
    probs = np.concatenate(probs, axis=0)
    df = df.iloc[args.seq_len-1:].copy()
    df["pred_up_prob"] = probs[:, 2]  # 類別2 = 上漲機率
    # df.to_csv("out.csv")
    print(df[["open_time", "close", "pred_up_prob"]].head())

if __name__ == "__main__":
    main()

# python .\load_model.py --weight_path "./logs/train_20251108_224758_BTCUSDT_ETHUSDT_BNBUSDT_DOGEUSDT_SOLUSDT/" --input_dim 24 --hidden 384 --layers 5 --dropout 0.3 --symbol BTCUSDT --year 2025 --month 8 --interval 10 --seq_len 36 --batch_size 1024
