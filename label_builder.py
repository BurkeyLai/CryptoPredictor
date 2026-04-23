#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Builder (daily-aware) for 1m OHLCV Parquet dataset.

Design goals:
  - **Time & IO are aligned with feature_builder.py** (same date partition logic).
  - Supports both layouts:
      A) Old monthly layout (hive partitions: exchange/symbol/year/month) -> writes monthly files
      B) New daily layout (hive partitions: symbol/year/month/date=YYYY-MM-DD.parquet) -> writes daily files
  - Generates 5 labels for supervised learning (H minutes horizon, default H=60):
      1) y_reg_ret_Hm   : future log return over H minutes
      2) y_cls_sign_Hm  : direction label with dead-zone epsilon
      3) y_tp_sl_Hm     : TP/SL first-hit (+1/-1/0) and y_tp_sl_Hm_t_hit
      4) y_tb_Hm        : triple-barrier (+1/-1/0) and y_tb_Hm_t_hit
      5) y_vol_Hm       : future realized volatility over H minutes

Example:
    python label_builder.py --src data_daily --dst labels_daily \
      --symbols BTCUSDT,ETHUSDT --start 2021-01-01 --end 2021-02-01 \
      --horizon_m 60 --epsilon_bp 5 --tp_bp 50 --sl_bp 35 \
      --tb_k_up 2.0 --tb_k_dn 2.0 --tb_vol_window_m 60 --tb_vol_method ewm
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ----------------------
# Args (mirrors feature_builder.py, plus label hyperparams)
# ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data_daily", help="Input root (Parquet dataset)")
    ap.add_argument("--dst", type=str, default="labels_daily", help="Output root (Parquet dataset)")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT", help="Comma-separated symbols")
    ap.add_argument("--start", type=str, default="2020-08-21", help="UTC start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default="2025-08-22", help="UTC end date (YYYY-MM-DD), exclusive")
    ap.add_argument("--enforce_continuous", action="store_true",
                    help="Reindex to continuous 1-minute timeline per symbol (forward-fill OHLC; volumes untouched)")
    # Label hyper-params
    ap.add_argument("--horizon_m", type=int, default=60, help="Horizon in minutes for all labels") # horizon_m: 預測視野 (單位: 分鐘)，用來定義標籤要觀察「未來 horizon_m 分鐘」的價格/報酬/方向。如果 interval ≠ 1 分鐘（例如 5 分鐘 K 線），那 horizon_m=60 就是「未來 60 分鐘 = 12 根 5 分鐘 K 線」。
    ap.add_argument("--epsilon_bp", type=float, default=5.0, help="Dead-zone for sign label, basis points")
    ap.add_argument("--tp_bp", type=float, default=50.0, help="TP in basis points (e.g., 50=0.50%)")
    ap.add_argument("--sl_bp", type=float, default=35.0, help="SL in basis points (e.g., 35=0.35%)")
    ap.add_argument("--tb_k_up", type=float, default=2.0, help="Triple-barrier up multiplier (× sigma)")
    ap.add_argument("--tb_k_dn", type=float, default=2.0, help="Triple-barrier down multiplier (× sigma)")
    ap.add_argument("--tb_vol_window_m", type=int, default=60, help="Window for sigma estimation (minutes)")
    ap.add_argument("--tb_vol_method", type=str, default="ewm", choices=["ewm","rolling"], help="Sigma method")
    ap.add_argument("--interval", type=int, default=1, help="Resample interval in minutes (e.g., 5 for 5-min K-lines)")
    ap.add_argument("--cls_type", type=str, default='atr', help="Classification type for labels, atr or quantile")
    return ap.parse_args()

# ----------------------
# Layout detection (same heuristic as feature_builder.py)
# ----------------------
def using_daily_partitions(dataset: ds.Dataset) -> bool:
    try:
        schema = dataset.schema
        return any(name == "date" for name in schema.names)
    except Exception:
        return False

# ----------------------
# Label primitives
# ----------------------
def add_logret_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_close"] = np.log(df["close"].replace(0, np.nan)).ffill().fillna(0.0)
    df["ret_1m"] = df["log_close"].diff()
    return df

def y_future_return(df: pd.DataFrame, H: int) -> pd.Series:
    return df["log_close"].shift(-H) - df["log_close"]

# 這個版本的 y_future_maxmin_return 是基於 log return 計算的，並且直接返回未來 H 分鐘內 log return 的極值。這樣做的好處是可以捕捉到未來價格變動的最大幅度，無論是上漲還是下跌，對於一些風險管理或波動率預測的任務可能更有用。
def y_future_maxmin_return(df: pd.DataFrame, H: int) -> pd.Series:
    future_max = df["close"].rolling(window=H, min_periods=1).max().shift(-H+1)
    future_min = df["close"].rolling(window=H, min_periods=1).min().shift(-H+1)
    ret1 = (future_max - df["close"]) / df["close"]
    ret2 = (future_min - df["close"]) / df["close"]
    
    return ret1.where(ret1.abs() > ret2.abs(), ret2)

def y_direction_with_deadzone(y_ret: pd.Series, epsilon: float) -> pd.Series:
    out = pd.Series(np.zeros(len(y_ret), dtype="int8"), index=y_ret.index)
    out = out.mask(y_ret >  epsilon,  1)
    out = out.mask(y_ret < -epsilon, -1)
    out = out.fillna(0).astype("int8")
    out = out.where(y_ret.notna(), np.nan)
    return out

# 二分類（只標記正/負，0 當作負或直接去除）
def y_direction_binary(y_ret: pd.Series, epsilon: float) -> pd.Series:
    out = pd.Series(np.zeros(len(y_ret), dtype="int8"), index=y_ret.index)
    out = out.mask(y_ret > epsilon, 1)
    out = out.mask(y_ret <= epsilon, 0)
    out = out.where(y_ret.notna(), np.nan)
    return out

def label_tp_sl(close: np.ndarray, high: np.ndarray, low: np.ndarray, H: int, tp: float, sl: float):
    n = len(close)
    y = np.full(n, np.nan, dtype="float32")
    t_hit = np.full(n, -1, dtype="int16")
    for i in range(0, max(0, n - H)):
        base = close[i]
        up = base * (1.0 + tp)
        dn = base * (1.0 + sl)
        hi = high[i+1:i+1+H]
        lo = low[i+1:i+1+H]
        hit_up = np.where(hi >= up)[0]
        hit_dn = np.where(lo <= dn)[0]
        iu = hit_up[0] if hit_up.size else np.inf
        idn = hit_dn[0] if hit_dn.size else np.inf
        if iu < idn:
            y[i] = 1.0
            t_hit[i] = int(iu + 1)
        elif idn < iu:
            y[i] = -1.0
            t_hit[i] = int(idn + 1)
        else:
            y[i] = 0.0
            t_hit[i] = -1
    return y, t_hit

def compute_sigma(ret_1m: pd.Series, window_m: int, method: str = "ewm") -> pd.Series:
    if method == "ewm":
        sigma = ret_1m.ewm(span=window_m, adjust=False).std(bias=False)
    else:
        sigma = ret_1m.rolling(window_m, min_periods=window_m).std()
    return sigma

def label_triple_barrier(df: pd.DataFrame, H: int, k_up: float, k_dn: float, vol_window_m: int, vol_method: str):
    n = len(df)
    y = np.full(n, np.nan, dtype="float32")
    t_hit = np.full(n, -1, dtype="int16")

    sigma_t = compute_sigma(df["ret_1m"], vol_window_m, vol_method).to_numpy()
    scale = np.sqrt(H)
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()

    for i in range(0, max(0, n - H)):
        s = sigma_t[i]
        if not np.isfinite(s) or s <= 0:
            y[i] = np.nan
            t_hit[i] = -1
            continue
        up = close[i] * (1.0 + k_up * s * scale)
        dn = close[i] * (1.0 - k_dn * s * scale)
        hi = high[i+1:i+1+H]
        lo = low[i+1:i+1+H]
        hit_up = np.where(hi >= up)[0]
        hit_dn = np.where(lo <= dn)[0]
        iu = hit_up[0] if hit_up.size else np.inf
        idn = hit_dn[0] if hit_dn.size else np.inf
        if iu < idn:
            y[i] = 1.0
            t_hit[i] = int(iu + 1)
        elif idn < iu:
            y[i] = -1.0
            t_hit[i] = int(idn + 1)
        else:
            y[i] = 0.0
            t_hit[i] = -1
    return y, t_hit

def future_realized_vol(ret_1m: pd.Series, H: int) -> pd.Series:
    return ret_1m.shift(-1).rolling(H, min_periods=H).std() * np.sqrt(H)

# ----------------------
# IO writers (mirrors feature_builder.py logic & time format)
# ----------------------
def write_out(df: pd.DataFrame, out_root: Path, daily_mode: bool, H: int, interval: int):
    if df.empty:
        return
    df = df.copy()
    if daily_mode:
        # IMPORTANT: identical to feature_builder.py (UTC+8 by shift and floor)
        df["date"] = (df["open_time"] - pd.Timedelta(hours=8)).dt.floor("D").dt.date.astype(str)
        df["year"] = df["open_time"].dt.year.astype("int16")
        df["month"] = df["open_time"].dt.month.astype("int8")
        for (sym, y, m, d), part in df.groupby(["symbol","year","month","date"], sort=True):
            outdir = Path(out_root) / f"{interval}min" / f"symbol={sym}"/ f"year={y}"/ f"month={m:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            fn = outdir / f"date={d}.parquet"
            part.sort_values("open_time").to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
            print(f"[write] {sym} {d} rows={len(part)} -> {fn}")
    else:
        if "exchange" not in df.columns:
            df["exchange"] = "binance"
        df["year"] = df["open_time"].dt.year.astype("int16")
        df["month"] = df["open_time"].dt.month.astype("int8")
        for (ex, sym, y, m), part in df.groupby(["exchange","symbol","year","month"], sort=True):
            outdir = Path(out_root) / f"exchange={ex}"/ f"{interval}min"/ f"symbol={sym}"/ f"year={y}"/ f"month={m:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            fn = outdir / f"labels-{y}-{m:02d}.parquet"
            part.sort_values("open_time").to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
            print(f"[write] {sym} {y}-{m:02d} rows={len(part)} -> {fn}")

# 可以複用 feature_builder.py 的 resample_ohlcv 函數
def resample_ohlcv(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Resample 1-min K-lines to N-min."""
    if interval <= 1:
        return df
    
    df = df.set_index("open_time")
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }
    
    agg_rules = {col: rule for col, rule in agg_dict.items() if col in df.columns}
    
    exchange = df["exchange"].iloc[0] if "exchange" in df.columns else None
    symbol = df["symbol"].iloc[0]
    
    resampled = df.resample(f"{interval}min", closed="left", label="left").agg(agg_rules)
    
    resampled = resampled.reset_index()
    if exchange is not None:
        resampled["exchange"] = exchange
    resampled["symbol"] = symbol
    
    return resampled

# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)

    dataset = ds.dataset(src, format="parquet", partitioning="hive")
    daily_mode = using_daily_partitions(dataset)
    print(f"[info] daily_mode={daily_mode} based on input dataset layout")

    eps = float(args.epsilon_bp) / 1e4
    tp = float(args.tp_bp) / 1e4
    sl = -float(args.sl_bp) / 1e4
    k_up = float(args.tb_k_up)
    k_dn = float(args.tb_k_dn)
    vol_w = int(args.tb_vol_window_m)
    vol_method = args.tb_vol_method

    # 調整預測視窗 (從分鐘轉換成 K 線數量)
    H = args.horizon_m // args.interval  # 例如 60分鐘 ÷ 5分鐘 = 12根K線
    if H == 0:
        raise ValueError(f"horizon_m ({args.horizon_m}) must be >= interval ({args.interval})")
    
    print(f"[info] horizon_m={args.horizon_m} -> {H} {args.interval}-min bars")

    for sym in symbols:
        print(f"Building labels for {sym} ...")
        tbl = dataset.to_table(filter=(ds.field("symbol") == sym))
        if tbl.num_rows == 0:
            print(f"  No rows for {sym}, skip.")
            continue
        cols = [c for c in ["exchange","symbol","open_time","open","high","low","close"] if c in tbl.schema.names]
        df = tbl.select(cols).to_pandas()
        df = df[(df["open_time"] >= start) & (df["open_time"] < end)].copy()
        if df.empty:
            print(f"  Empty range for {sym}, skip.")
            continue
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 在 enforce_continuous 之前先做重採樣
        df = resample_ohlcv(df, args.interval)

        if args.enforce_continuous:
            full_idx = pd.date_range(
                df["open_time"].iloc[0], 
                df["open_time"].iloc[-1], 
                freq=f"{args.interval}min", 
                tz="UTC"
            )
            df = df.set_index("open_time").reindex(full_idx)
            df.index.name = "open_time"
            for col in ["open","high","low","close"]:
                df[col] = df[col].ffill()
            if "exchange" in df.columns:
                df["exchange"] = df["exchange"].ffill()
            df["symbol"] = df["symbol"].ffill()
            df = df.reset_index().rename(columns={"index":"open_time"})

        # Base returns
        df = add_logret_1m(df)

        # 1) future return (用調整後的 H)
        if H != 1:
            y_ret = y_future_maxmin_return(df, H)
        else:
            y_ret = y_future_return(df, H)
        # y_ret.to_csv("y_ret.csv")

        df[f"y_reg_ret_{args.horizon_m}m"] = y_ret * 100.0  # convert to percentage
        # df[f"y_reg_ret_{args.horizon_m}m"] = y_ret
        # 2) sign with dead-zone
        # df[f"y_cls_sign_{args.horizon_m}m"] = y_direction_binary(y_ret, eps)
        if args.cls_type == 'atr':
            # ====== ATR-based deadzone ======
            def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
                prev_close = close.shift(1)
                tr = pd.concat([
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()
                ], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
                return atr

            # 使用 atr_window 根 k 的 ATR 當噪音尺度
            atr_window = 14
            df[f"atr_{atr_window}"] = atr_wilder(df["high"], df["low"], df["close"], atr_window)

            # k = 幅度倍數（建議 0.5 ~ 1.0）
            k = 0.6
            # eps = k × ATR（未來報酬是 log return，所以要除以 price）
            # 避免 log return 與 ATR 量級不一致
            # y_cls_sign 的正負，其實是「報酬是否顯著大於當下噪音」
            # 2 = 正向突破；1 = 無趨勢；0 = 負向突破
            eps_series = k * df[f"atr_{atr_window}"] / df["close"]
            df[f"y_cls_sign_{args.horizon_m}m"] = np.where(
                y_ret > eps_series, 2,
                np.where(y_ret < -eps_series, 0, 1)
            )
        else:
            low = y_ret.quantile(0.3)
            high = y_ret.quantile(0.7)

            df[f"y_cls_sign_{args.horizon_m}m"] = np.where(
                y_ret > high, 2,
                np.where(y_ret < low, 0, 1)
            )

        dist = df[f"y_cls_sign_{args.horizon_m}m"].value_counts().reindex([0,1,2], fill_value=0)
        ratio = df[f"y_cls_sign_{args.horizon_m}m"].value_counts(normalize=True).reindex([0,1,2], fill_value=0)

        # print(dist)
        print(f"  Class distribution for y_cls_sign_{args.horizon_m}m:")
        print(ratio)

        # 3) TP/SL first hit (+ t_hit) (用調整後的 H)
        y_tpsl, t_tpsl = label_tp_sl(
            close=df["close"].to_numpy(),
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            H=H, tp=tp, sl=sl
        )
        df[f"y_tp_sl_{args.horizon_m}m"] = y_tpsl
        df[f"y_tp_sl_{args.horizon_m}m_t_hit"] = t_tpsl

        # 4) Triple barrier (+ t_hit) (用調整後的 H)
        y_tb, t_tb = label_triple_barrier(df, H, k_up, k_dn, vol_w, vol_method)
        df[f"y_tb_{args.horizon_m}m"] = y_tb
        df[f"y_tb_{args.horizon_m}m_t_hit"] = t_tb

        # 5) future realized vol (用調整後的 H)
        # t+1 ~ t+H 的 realized volatility
        df[f"y_vol_{args.horizon_m}m"] = future_realized_vol(df["ret_1m"], H)

        # Keep only rows with complete future window for the daily layout
        valid = df[f"y_reg_ret_{args.horizon_m}m"].notna()
        out_cols = ["symbol","open_time","close",
                    f"y_reg_ret_{args.horizon_m}m", f"y_cls_sign_{args.horizon_m}m",
                    f"y_tp_sl_{args.horizon_m}m", f"y_tp_sl_{args.horizon_m}m_t_hit",
                    f"y_tb_{args.horizon_m}m", f"y_tb_{args.horizon_m}m_t_hit",
                    f"y_vol_{args.horizon_m}m"]
        write_out(df.loc[valid, out_cols], dst, daily_mode, args.horizon_m, args.interval)

        print(f"  Done: {sym}")
        # break

if __name__ == "__main__":
    main()

# python label_builder.py --src data --dst labels --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --interval 10 --enforce_continuous

# 過去 k 棒顆粒度 30 分，且預測視窗為 120 分鐘
#python label_builder.py --src data --dst labels --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --horizon_m 120 --interval 30 --enforce_continuous
