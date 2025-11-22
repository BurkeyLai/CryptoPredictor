#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Builder (daily-aware) for 1m OHLCV Parquet dataset.

Supports both layouts:
  A) Old monthly layout (hive partitions: exchange/symbol/year/month)
  B) New daily layout (hive partitions: symbol/year/month/date=YYYY-MM-DD.parquet)

It computes:
  - SMA: 5/10/20/50/100/200
  - RSI(14) (Wilder)
  - ATR(14) (Wilder)
  - Realized Vol: rv_{5,15,30,60,120,240} based on 1m log returns
  - Volume & Trades z-scores over windows {30,60,120}

Output mirrors the input layout:
  - If input has date= partitions, output writes to features_daily/symbol=.../year=.../month=.../date=...
  - Otherwise, outputs to features/exchange=.../symbol=.../year=.../month=...

Example:
    python feature_builder.py --src data_daily --dst features_daily --symbols BTCUSDT,ETHUSDT --start 2021-01-01 --end 2021-02-01
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data_daily", help="Input root (Parquet dataset)")
    ap.add_argument("--dst", type=str, default="features_daily", help="Output root (Parquet dataset)")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT", help="Comma-separated symbols")
    ap.add_argument("--start", type=str, default="2020-08-21", help="UTC start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default="2025-08-22", help="UTC end date (YYYY-MM-DD), exclusive")
    ap.add_argument("--enforce_continuous", action="store_true",
                    help="Reindex to continuous 1-minute timeline per symbol (forward-fill OHLC, volume=0 for gaps)")
    ap.add_argument("--horizon_m", type=int, default=60, help="Horizon in minutes for all labels")
    ap.add_argument("--interval", type=int, default=1, help="Resample interval in minutes (e.g., 5 for 5-min K-lines)")
    return ap.parse_args()

# ----- Technicals -----
def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    return atr

def realized_vol(logret: pd.Series, windows=(5, 15, 30, 60, 120, 240)) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"rv_{w}"] = logret.rolling(w, min_periods=w).std() * np.sqrt(w)
    return pd.DataFrame(out)

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)

# ----- IO helpers -----
def using_daily_partitions(dataset: ds.Dataset) -> bool:
    # Heuristic: check for 'date' partition field in schema or file paths
    try:
        schema = dataset.schema
        # When scanning parquet with hive partitioning, partition fields show up as normal fields.
        return any(name == "date" for name in schema.names)
    except Exception:
        return False

def write_out(df: pd.DataFrame, out_root: Path, daily_mode: bool, interval: int):
    if df.empty:
        return
    df = df.copy()
    if daily_mode:
        # derive date from open_time in UTC+8 alignment by shifting -8h then flooring to day
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
        # monthly layout with exchange
        if "exchange" not in df.columns:
            df["exchange"] = "binance"
        df["year"] = df["open_time"].dt.year.astype("int16")
        df["month"] = df["open_time"].dt.month.astype("int8")
        for (ex, sym, y, m), part in df.groupby(["exchange","symbol","year","month"], sort=True):
            outdir = Path(out_root) / f"exchange={ex}"/ f"{interval}min"/ f"symbol={sym}"/ f"year={y}"/ f"month={m:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            fn = outdir / f"features-{y}-{m:02d}.parquet"
            part.sort_values("open_time").to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
            print(f"[write] {sym} {y}-{m:02d} rows={len(part)} -> {fn}")

def resample_ohlcv(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Resample 1-min K-lines to N-min."""
    if interval <= 1:
        return df
    
    # 確保時間索引
    df = df.set_index("open_time")
    
    # 定義重採樣規則
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_asset_volume": "sum",
        "number_of_trades": "sum"
    }
    
    # 只聚合存在的欄位
    agg_rules = {col: rule for col, rule in agg_dict.items() if col in df.columns}
    
    # 如果有 exchange/symbol，先記住
    exchange = df["exchange"].iloc[0] if "exchange" in df.columns else None
    symbol = df["symbol"].iloc[0]
    
    # 重採樣
    resampled = df.resample(f"{interval}min", closed="left", label="left").agg(agg_rules)
    
    # 重設索引並加回 exchange/symbol
    resampled = resampled.reset_index()
    if exchange is not None:
        resampled["exchange"] = exchange
    resampled["symbol"] = symbol
    
    return resampled

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

    # 調整預測視窗 (從分鐘轉換成 K 線數量)
    H = args.horizon_m // args.interval  # 例如 60分鐘 ÷ 5分鐘 = 12根K線
    if H == 0:
        raise ValueError(f"horizon_m ({args.horizon_m}) must be >= interval ({args.interval})")
    
    print(f"[info] horizon_m={args.horizon_m} -> {H} {args.interval}-min bars")
    
    for sym in symbols:
        print(f"Building features for {sym} ...")
        tbl = dataset.to_table(filter=(ds.field("symbol") == sym))
        if tbl.num_rows == 0:
            print(f"  No rows for {sym}, skip.")
            continue
        cols = [c for c in ["exchange","symbol","open_time","open","high","low","close","volume",
                            "quote_asset_volume","number_of_trades"] if c in tbl.schema.names]
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
            for col in ["volume","quote_asset_volume","number_of_trades"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            if "exchange" in df.columns:
                df["exchange"] = df["exchange"].ffill()
            df["symbol"] = df["symbol"].ffill()
            df = df.reset_index().rename(columns={"index":"open_time"})

        # Log returns
        df["log_close"] = np.log(df["close"].replace(0, np.nan)).ffill().fillna(0.0)
        df["ret_1m"] = df["log_close"].diff().fillna(0.0)

        # SMAs
        for w in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{w}"] = df["close"].rolling(w, min_periods=w).mean()

        # Normalized close
        for w in [5, 10, 20, 50, 100, 200]:
            df[f"close_norm_{w}"] = df["close"] / df[f"sma_{w}"] - 1.0

        # RSI & ATR
        df["rsi_14"] = rsi_wilder(df["close"], 14)
        df["atr_14"] = atr_wilder(df["high"], df["low"], df["close"], 14)

        # ===== True Range =====
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs()
            )
        )

        # ===== ATR for multiple horizons (including horizon_m) =====
        # convert horizon minutes → ATR bars
        atr_window = max(1, H)
        df[f"atr_{args.horizon_m}"] = df["tr"].rolling(atr_window).mean()

        # Realized vol
        rv = realized_vol(df["ret_1m"], windows=(5, 15, 30, 60, 120, 240))
        df = pd.concat([df, rv], axis=1)

        # Volume anomalies
        for w in [30, 60, 120]:
            if "volume" in df.columns:
                df[f"vol_z_{w}"] = zscore(df["volume"], w)
            if "number_of_trades" in df.columns:
                df[f"trades_z_{w}"] = zscore(df["number_of_trades"].astype(float), w)

        base_cols = ["symbol","open_time","close"]
        if "volume" in df.columns: base_cols.append("volume")
        if "number_of_trades" in df.columns: base_cols.append("number_of_trades")

        out_cols = base_cols + ["ret_1m","rsi_14","atr_14"] + [c for c in df.columns if c.startswith("sma_")] + [c for c in df.columns if c.startswith("rv_")] + [c for c in df.columns if c.endswith("_z_30") or c.endswith("_z_60") or c.endswith("_z_120")]
        df.to_csv('out.csv', index=False)
        # write_out(df[out_cols], dst, daily_mode, args.interval)
        print(f"  Done: {sym}")

if __name__ == "__main__":
    main()

# python feature_builder.py --src data --dst features --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --interval 10 --enforce_continuous

# 過去 k 棒顆粒度 30 分，且預測視窗為 120 分鐘(label_builder.py 有使用 ATR 也要對應修改)
# python feature_builder.py --src data --dst features --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --horizon_m 120 --interval 30 --enforce_continuous