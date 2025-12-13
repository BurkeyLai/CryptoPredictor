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

    # --- (A) load reference cross-asset (BTC) for multi-asset context ---
    # build a small reference df for BTCUSDT aligned to same interval
    ref_symbols = {"BTCUSDT": None}
    try:
        tbl_btc = dataset.to_table(filter=(ds.field("symbol") == "BTCUSDT"))
        if tbl_btc.num_rows:
            df_btc = tbl_btc.select(["open_time","close"]).to_pandas()
            df_btc = df_btc[(df_btc["open_time"] >= start) & (df_btc["open_time"] < end)].copy()
            df_btc.sort_values("open_time", inplace=True)
            df_btc = resample_ohlcv(df_btc, args.interval)
            df_btc["log_close"] = np.log(df_btc["close"].replace(0, np.nan)).ffill().fillna(0.0)
            df_btc["ret_1m_btc"] = df_btc["log_close"].diff().fillna(0.0)
            # compute some rv_60 for BTC as context
            df_btc["rv_60_btc"] = df_btc["ret_1m_btc"].rolling(60//args.interval, min_periods=60//args.interval).std() * np.sqrt(60//args.interval)
            ref_symbols["BTCUSDT"] = df_btc[["open_time","ret_1m_btc","rv_60_btc"]].set_index("open_time")
            print("[info] built BTC context")
    except Exception:
        ref_symbols["BTCUSDT"] = None

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

        # --- Multi-window returns & EMA slopes ---
        # bars based on args.interval
        b1 = 1
        b3 = max(1, 3 * (1))    # 3 bars (if using 1-min base), keep small; or use multiples of interval.
        # better: use number of bars for 1,3,6,12 *in current resolution*
        r_bars = [3, 6, 12]   # interpret as N bars
        for n in r_bars:
            df[f"ret_{n}m"] = df["log_close"].diff(n).fillna(0.0)   # log return over n bars

        # EMAs: periods in bars (12,24,48)
        for p in [12, 24, 48]:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
            # slope = difference over p bars normalized by p (gives per-bar slope)
            # df[f"ema_{p}_slope"] = (df[f"ema_{p}"] - df[f"ema_{p}"].shift(p)) / p
            df[f"ema_{p}_slope"] = (np.log(df[f"ema_{p}"]) - np.log(df[f"ema_{p}"].shift(p))) / p
            # deviation (乖離)
            df[f"ema_{p}_dev"] = (df["close"] / df[f"ema_{p}"] - 1.0).fillna(0.0)

        # SMAs
        for w in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{w}"] = df["close"].rolling(w, min_periods=w).mean()

        # Normalized close
        for w in [5, 10, 20, 50, 100, 200]:
            df[f"close_norm_{w}"] = df["close"] / df[f"sma_{w}"] - 1.0

        # --- Bollinger Bands (20,2) and position; Donchian channel and breakout distance ---
        bb_w = 20
        bb_k = 2
        df["bb_mid"] = df["close"].rolling(bb_w, min_periods=bb_w).mean()
        df["bb_std"] = df["close"].rolling(bb_w, min_periods=bb_w).std()
        df["bb_upper"] = df["bb_mid"] + bb_k * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - bb_k * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]      # relative width
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pos"] = df["bb_pos"].fillna(0.5)

        # Donchian (lookback = 20)
        don_w = 20
        df["don_high"] = df["high"].rolling(don_w, min_periods=don_w).max().shift(1)
        df["don_low"] = df["low"].rolling(don_w, min_periods=don_w).min().shift(1)
        df["don_width"] = (df["don_high"] - df["don_low"]) / ((df["don_high"] + df["don_low"]) / 2).replace(0, np.nan)
        # breakout distance (positive if above upper; negative if below lower)
        df["don_break_dist"] = np.where(
            df["close"] > df["don_high"], 
            (df["close"] - df["don_high"]) / df["don_high"],
            np.where(
                df["close"] < df["don_low"], 
                (df["don_low"] - df["close"]) / df["don_low"], 
                0.0
            )
        )

        # bars since last breakout (upper or lower)
        # mask_break = (df["close"] > df["don_high"].shift(1)) | (df["close"] < df["don_low"].shift(1))
        # compute distance since last True: cumulative count reset
        # df["since_break"] = (~mask_break).astype(int).groupby((mask_break != mask_break.shift(1)).cumsum()).cumsum()
        # simpler fallback: compute last index of True using forward/backfill (expensive but OK for monthly)
        # last_idx = mask_break[::-1].cumsum()[::-1]  # not exact bars since; leave as heuristic if needed
        # 注意：since_break 這裡給出一個輕量 heuristic；若要精確 bars-since，建議用 np.where loop 或 pd.Series.where(...).ffill() 模式（可再補）。

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

        # --- extra realized volatility measures (Parkinson, Rogers-Satchell) ---
        # Parkinson (uses high/low)
        def parkinson_vol(high, low, window):
            # Parkinson estimator: sqrt( (1/(4n ln2)) * sum( (ln(high/low))^2 ) )
            term = (np.log(high / low).replace([np.inf, -np.inf], np.nan))**2
            denom = 4.0 * window * np.log(2.0)
            return np.sqrt(term.rolling(window, min_periods=window).sum() / denom)

        # Rogers-Satchell (uses open, high, low, close)
        def rogers_satchell(open, high, low, close, window):
            rs = (np.log(high / close).replace([np.inf,-np.inf],0) * np.log(high / open).replace([np.inf,-np.inf],0) +
                np.log(low / close).replace([np.inf,-np.inf],0) * np.log(low / open).replace([np.inf,-np.inf],0))
            return np.sqrt(rs.rolling(window, min_periods=window).mean().abs())

        for w in [5, 15, 30, 60]:
            df[f"park_{w}"] = parkinson_vol(df["high"], df["low"], w).fillna(0.0)
            df[f"rs_{w}"] = rogers_satchell(df["open"], df["high"], df["low"], df["close"], w).fillna(0.0)

        # realized var windows already present as rv_*; add percentile of recent volatility
        for w in [60,120,240]:
            df[f"rv_{w}_pctile"] = df[f"rv_{w}"].rolling(100, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>0 else 0.5)

        # Volume anomalies
        for w in [30, 60, 120]:
            if "volume" in df.columns:
                df[f"vol_z_{w}"] = zscore(df["volume"], w)
            if "number_of_trades" in df.columns:
                df[f"trades_z_{w}"] = zscore(df["number_of_trades"].astype(float), w)

        # --- Volume & money flow features ---
        # Volume ROC (percent change over window)
        for w in [5, 15, 60]:
            if "volume" in df.columns:
                df[f"vol_roc_{w}"] = df["volume"].pct_change(w).fillna(0.0)

        # OBV
        def obv(series_close, series_vol):
            sign = np.sign(series_close.diff()).ffill().fillna(0)
            return (sign * series_vol).fillna(0.0).cumsum()
        if "volume" in df.columns:
            df["obv"] = obv(df["close"], df["volume"])
            df["obv_slope_60"] = (df["obv"] - df["obv"].shift(60//args.interval)).fillna(0.0) / max(1, 60//args.interval)

        # Chaikin Money Flow (CMF) over 20 windows
        if "volume" in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3.0
            mf = ((tp - df["low"]) - (df["high"] - tp)) / (df["high"] - df["low"]).replace(0, np.nan) * df["volume"]
            df["cmf_20"] = mf.rolling(20, min_periods=1).sum() / df["volume"].rolling(20, min_periods=1).sum().replace(0, np.nan).fillna(0.0)

        # MFI (Money Flow Index) 14
        if "volume" in df.columns:
            up = tp.diff() > 0
            positive_mf = (tp * df["volume"]).where(up, 0.0).rolling(14, min_periods=1).sum()
            negative_mf = (tp * df["volume"]).where(~up, 0.0).rolling(14, min_periods=1).sum()
            df["mfi_14"] = 100 - 100 / (1 + positive_mf / negative_mf.replace(0, 1e-8))

        # VWAP deviation: rolling VWAP over 1 day or window
        if "volume" in df.columns:
            vw_window = max(1, 24*60 // args.interval)  # ~ one day
            cum_vp = ( (df["close"] * df["volume"]).rolling(vw_window, min_periods=1).sum() )
            cum_v = df["volume"].rolling(vw_window, min_periods=1).sum().replace(0, np.nan)
            df["vwap_1d"] = cum_vp / cum_v
            df["close_vwap_dev"] = df["close"] / df["vwap_1d"] - 1.0

        # volume-price divergence flag: correlation between volume and returns in short window
        def rolling_corr(a, b, window):
            return a.rolling(window, min_periods=1).corr(b)
        if "volume" in df.columns:
            df["vol_ret_corr_30"] = rolling_corr(df["volume"], df["ret_1m"], 30)
            df["vol_price_divergence"] = (df["vol_ret_corr_30"] < -0.3).astype(int)

        # --- seasonality / time-of-day features ---
        df["hour"] = df["open_time"].dt.hour
        df["dow"] = df["open_time"].dt.dayofweek
        # sin/cos encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        # optional one-hot for dow (7 cols) if desired
        for i in range(7):
            df[f"dow_{i}"] = (df["dow"] == i).astype(int)

        # --- cross-asset: merge BTC context if available and current symbol != BTC ---
        if ref_symbols.get("BTCUSDT") is not None and sym != "BTCUSDT":
            try:
                # align on open_time
                df = df.set_index("open_time")
                btc_ctx = ref_symbols["BTCUSDT"]
                btc_ctx = btc_ctx.shift(1)
                df = df.join(btc_ctx, how="left")
                df = df.reset_index().rename(columns={"index":"open_time"})
                # rename merged columns
                if "ret_1m_btc" in df.columns:
                    df.rename(columns={"ret_1m_btc":"ret_1m_btc_ctx"}, inplace=True)
                if "rv_60_btc" in df.columns:
                    df.rename(columns={"rv_60_btc":"rv_60_btc_ctx"}, inplace=True)
            except Exception:
                pass
        else:
            # for BTC itself, fill zeros for ctx
            df["ret_1m_btc_ctx"] = 0.0
            df["rv_60_btc_ctx"] = 0.0
        """
        # --- simple redundancy removal + feature importance hint (mutual info) ---
        from sklearn.feature_selection import mutual_info_classif

        # candidate numeric cols (exclude meta)
        num_cols = [c for c in df.columns if c not in ("symbol","open_time","date","year","month","exchange") and pd.api.types.is_numeric_dtype(df[c])]
        # compute correlation matrix and drop one of highly correlated pairs
        corr = df[num_cols].corr().abs()
        to_drop = set()
        threshold = 0.95
        for i, c1 in enumerate(corr.columns):
            if c1 in to_drop: continue
            for c2 in corr.columns[i+1:]:
                if corr.loc[c1,c2] > threshold:
                    to_drop.add(c2)   # drop c2 as duplicate of c1

        to_keep = [c for c in num_cols if c not in to_drop]
        print(f"[feature_sel] dropping {len(to_drop)} cols due to high corr")

        # mutual info (requires target; here compute against next-interval sign proxy if exists)
        if f"y_cls_sign_{args.horizon_m}m" in df.columns:
            try:
                # dropna and small sample
                df_mi = df.dropna(subset=to_keep + [f"y_cls_sign_{args.horizon_m}m"]).tail(20000)
                Xmi = df_mi[to_keep].fillna(0.0).values
                ymi = df_mi[f"y_cls_sign_{args.horizon_m}m"].astype(int).values
                mi = mutual_info_classif(Xmi, ymi, discrete_features=False, random_state=0)
                mi_series = pd.Series(mi, index=to_keep).sort_values(ascending=False)
                print("[feature_sel] top features by mutual info:\n", mi_series.head(20))
            except Exception:
                pass
        """
        # final out_cols can use to_keep (or further user selection)
        base_cols = ["symbol","open_time","close"]
        if "volume" in df.columns: base_cols.append("volume")
        if "number_of_trades" in df.columns: base_cols.append("number_of_trades")

        # out_cols = base_cols + \
        #     ["ret_1m","rsi_14","atr_14"] + \
        #     [c for c in df.columns if c.startswith("sma_")] + \
        #     [c for c in df.columns if c.startswith("rv_")] + \
        #     [c for c in df.columns if c.endswith("_z_30") or c.endswith("_z_60") or c.endswith("_z_120")]
        out_cols = (
            base_cols
            # --- returns / trend ---
            # + ["ret_1m"]
            + [c for c in df.columns if c.startswith("ret_")]

            # --- moving averages / trend shape ---
            + [c for c in df.columns if c.startswith("ema_")]
            + [c for c in df.columns if c.startswith("sma_")]
            + [c for c in df.columns if c.startswith("close_norm_")]

            # --- momentum / oscillators ---
            + ["rsi_14"]

            # --- volatility ---
            + ["atr_14"]
            + ["tr"]
            + [f"atr_{args.horizon_m}"]
            + [c for c in df.columns if c.startswith("rv_")]
            + [c for c in df.columns if c.startswith("park_")]
            + [c for c in df.columns if c.startswith("rs_")]

            # --- Bollinger / Donchian ---
            + [c for c in df.columns if c.startswith("bb_")]
            + [c for c in df.columns if c.startswith("don_")]
            # + ["since_break"]

            # --- volume & money flow ---
            + [c for c in df.columns if c.startswith("vol_roc_")]
            + ["obv", "obv_slope_60"]
            + ["cmf_20", "mfi_14"]
            + ["close_vwap_dev"]
            + ["vol_ret_corr_30", "vol_price_divergence"]

            # --- volume / trades z-score（你原本的） ---
            + [c for c in df.columns if c.endswith("_z_30") or c.endswith("_z_60") or c.endswith("_z_120")]

            # --- seasonality ---
            + ["hour_sin", "hour_cos"]
            + [c for c in df.columns if c.startswith("dow_")]

            # --- cross-asset context ---
            + ["ret_1m_btc_ctx", "rv_60_btc_ctx"]
        )
        # print(f"  Output columns: {out_cols}")
        for col in out_cols:
            if col not in df.columns:
                print(f"  [warn] missing expected out_col: {col}")

        df.to_csv('out.csv', index=False)
        # write_out(df[out_cols], dst, daily_mode, args.interval)
        print(f"  Done: {sym}")
        # break

if __name__ == "__main__":
    main()

# python feature_builder.py --src data --dst features --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --interval 10 --enforce_continuous

# 過去 k 棒顆粒度 30 分，且預測視窗為 120 分鐘(label_builder.py 有使用 ATR 也要對應修改)
# python feature_builder.py --src data --dst features --symbols BTCUSDT,BNBUSDT,DOGEUSDT,ETHUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --horizon_m 120 --interval 30 --enforce_continuous