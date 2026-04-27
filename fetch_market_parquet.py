#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified downloader for multiple markets -> Parquet (hive partitions).

Markets
-------
- crypto : Binance spot 1-minute klines (OHLCV)
- stocks : Yahoo Finance daily bars via yfinance (US + TW)

Why a single entry?
------------------
This project originally had:
- fetch_klines_parquet.py (crypto)
- fetch_yfinance_daily_parquet.py (stocks)
This script merges both so you can switch by parameters.

Layouts
-------
Crypto (1m, Taipei-day files):
  out/
    symbol=BTCUSDT/year=2025/month=03/date=2025-03-21.parquet

Stocks (1d, written under 1440min/ so feature_builder/label_builder can use --interval 1440):
  out/
    1440min/symbol=2330.TW/year=2025/month=03/date=2025-03-21.parquet
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# -----------------------
# Common helpers
# -----------------------
TPE = timezone(timedelta(hours=8))


def _norm_symbols(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _find_latest_date_parquet(base: Path) -> str | None:
    """
    Find latest YYYY-MM-DD among files named date=YYYY-MM-DD.parquet under base (recursive).
    """
    if not base.exists():
        return None
    latest = None
    for p in base.rglob("date=*.parquet"):
        name = p.name
        if not (name.startswith("date=") and name.endswith(".parquet")):
            continue
        d = name[len("date=") : -len(".parquet")]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) is None:
            continue
        if (latest is None) or (d > latest):
            latest = d
    return latest


# -----------------------
# Crypto: Binance 1m
# -----------------------
BINANCE_API = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVAL = "1m"
BINANCE_LIMIT = 1000


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _binance_fetch_range(
    symbol: str,
    start: datetime,
    end: datetime,
    session: requests.Session,
    retry: int = 5,
    cooldown: float = 1.0,
):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(minutes=BINANCE_LIMIT), end)
        params = {
            "symbol": symbol,
            "interval": BINANCE_INTERVAL,
            "startTime": _to_ms(cur),
            "endTime": _to_ms(nxt) - 1,
            "limit": BINANCE_LIMIT,
        }
        for attempt in range(retry):
            try:
                r = session.get(BINANCE_API, params=params, timeout=30)
                if r.status_code in (418, 451):
                    raise RuntimeError(f"HTTP {r.status_code} (IP banned/geo blocked).")
                if r.status_code == 429:
                    sleep_t = cooldown * (2**attempt)
                    print(f"[429] throttled; sleeping {sleep_t:.1f}s")
                    time.sleep(sleep_t)
                    continue
                r.raise_for_status()
                data = r.json()
                if not data:
                    print(f"[empty] {symbol} {cur:%Y-%m-%d %H:%M}..{nxt:%Y-%m-%d %H:%M}")
                    break
                cols = [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ]
                df = pd.DataFrame(data, columns=cols)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                num_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_asset_volume",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                ]
                df[num_cols] = df[num_cols].astype(float)
                df["number_of_trades"] = df["number_of_trades"].astype("int32")
                df["symbol"] = symbol
                print(f"[fetch] {symbol} {cur:%Y-%m-%d %H:%M}..{nxt:%Y-%m-%d %H:%M} -> {len(df)} rows")
                yield df[
                    [
                        "symbol",
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                    ]
                ]
                break
            except Exception as e:
                if attempt == retry - 1:
                    raise
                sleep_t = cooldown * (2**attempt)
                print(f"[error] {type(e).__name__}: {e}; retry in {sleep_t:.1f}s")
                time.sleep(sleep_t)
        cur = nxt


def _daterange_days_tpe(start: datetime, end: datetime):
    start_tpe = start.astimezone(TPE).date()
    end_tpe = end.astimezone(TPE).date()
    cur = datetime.combine(start_tpe, datetime.min.time()).replace(tzinfo=TPE)
    while cur.date() < end_tpe:
        yield cur
        cur += timedelta(days=1)


def _crypto_find_last_tpe_date(out_root: Path, symbol: str) -> str | None:
    base = out_root / f"symbol={symbol}"
    return _find_latest_date_parquet(base)


def _crypto_write_day(symbol: str, day_tpe: datetime, out_root: Path) -> int:
    start_tpe = day_tpe.replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=TPE)
    end_tpe = start_tpe + timedelta(days=1)
    start_utc = start_tpe.astimezone(timezone.utc)
    end_utc = end_tpe.astimezone(timezone.utc)

    parts: list[pd.DataFrame] = []
    with requests.Session() as sess:
        for df in _binance_fetch_range(symbol, start_utc, end_utc, sess):
            parts.append(df)
    if not parts:
        print(f"[skip] {symbol} {day_tpe.date()} (no data)")
        return 0

    big = pd.concat(parts, ignore_index=True).sort_values("open_time")
    big = big[(big["open_time"] >= start_utc) & (big["open_time"] < end_utc)].reset_index(drop=True)

    y, m = day_tpe.year, day_tpe.month
    outdir = out_root / f"symbol={symbol}" / f"year={y}" / f"month={m:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"date={day_tpe.date()}.parquet"
    big.to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
    print(f"[write] {symbol} {day_tpe.date()} rows={len(big)} -> {fn}")
    return len(big)


def cmd_crypto(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Crypto: Binance 1m klines -> Parquet (Taipei-day)")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT")
    ap.add_argument("--start", type=str, default="2020-08-21", help="UTC date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default="2025-08-21", help="UTC date (YYYY-MM-DD, exclusive)")
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--auto_resume", action="store_true", help="Start from (last existing TPE date - 1 day) per symbol")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    user_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"End (UTC): {end} (exclusive) | auto_resume={args.auto_resume}")
    for sym in symbols:
        start = user_start
        if args.auto_resume:
            last = _crypto_find_last_tpe_date(out_root, sym)
            if last:
                last_dt = datetime.fromisoformat(last).replace(tzinfo=TPE)
                resume_dt = last_dt - timedelta(days=1)
                start = resume_dt.astimezone(timezone.utc)
                print(f"[resume] {sym}: found last TPE date={last} -> resume from previous day {resume_dt.date()} (UTC start {start.date()})")
            else:
                print(f"[resume] {sym}: no existing files, use user --start {start.date()}")

        print(f"=== {sym} === Date range start={start} end={end}")
        total = 0
        for day_tpe in _daterange_days_tpe(start, end):
            total += _crypto_write_day(sym, day_tpe, out_root)
        print(f"[done] {sym} total rows={total}")
    return 0


# -----------------------
# Stocks: Yahoo via yfinance
# -----------------------
def _is_numeric_ticker(sym: str) -> bool:
    return re.fullmatch(r"\d{4,6}", sym) is not None


def _apply_tw_suffix(sym: str, tw_suffix: str) -> str:
    if tw_suffix == "none":
        return sym
    if "." in sym:
        return sym
    if _is_numeric_ticker(sym):
        return f"{sym}{tw_suffix}"
    return sym


def _stocks_find_last_date(out_root: Path, symbol: str) -> str | None:
    base = out_root / "1440min" / f"symbol={symbol}"
    return _find_latest_date_parquet(base)

def _stocks_find_last_open_time(out_root: Path, symbol: str, storage: str) -> pd.Timestamp | None:
    """
    Return latest open_time (UTC) we have on disk for the given symbol.
    Supports:
    - daily  : date=YYYY-MM-DD.parquet
    - monthly: part-YYYY-MM.parquet
    - yearly : part-YYYY.parquet
    """
    storage = storage.lower()
    base = out_root / "1440min" / f"symbol={symbol}"
    if not base.exists():
        return None

    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None  # type: ignore[assignment]

    cand: list[Path] = []
    if storage == "daily":
        cand = list(base.rglob("date=*.parquet"))
    elif storage == "monthly":
        cand = list(base.rglob("part-????-??.parquet"))
    elif storage == "yearly":
        cand = list(base.rglob("part-????.parquet"))
    else:
        return None

    if not cand:
        return None

    # prefer newest by filename sort; then read max(open_time) from that file
    cand_sorted = sorted(cand, key=lambda p: p.as_posix())
    last_file = cand_sorted[-1]

    if pq is None:
        # fallback: load with pandas (slower)
        df = pd.read_parquet(last_file, columns=["open_time"])
        if df.empty:
            return None
        return pd.to_datetime(df["open_time"], utc=True).max()

    table = pq.read_table(last_file, columns=["open_time"])
    if table.num_rows == 0:
        return None
    s = table.column(0).to_pandas()
    ts = pd.to_datetime(s, utc=True).max()
    return ts


def _stocks_to_daily_bars(symbol: str, hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    df = hist.copy()
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in required):
        missing = [k for k in required if k not in cols]
        raise ValueError(f"Missing columns from yfinance history: {missing}; got={list(df.columns)}")
    df = df.rename(
        columns={
            cols["open"]: "open",
            cols["high"]: "high",
            cols["low"]: "low",
            cols["close"]: "close",
            cols["volume"]: "volume",
        }
    )
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        open_time = idx.tz_localize("UTC")
    else:
        open_time = idx.tz_convert("UTC")
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "open_time": open_time,
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        }
    )
    return out.dropna(subset=["open_time", "open", "high", "low", "close"]).sort_values("open_time").reset_index(drop=True)


def _stocks_write_daily_partitions(df: pd.DataFrame, out_root: Path) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = (df["open_time"] - pd.Timedelta(hours=8)).dt.floor("D").dt.date.astype(str)
    df["year"] = df["open_time"].dt.year.astype("int16")
    df["month"] = df["open_time"].dt.month.astype("int8")
    total = 0
    for (sym, y, m, d), part in df.groupby(["symbol", "year", "month", "date"], sort=True):
        outdir = out_root / "1440min" / f"symbol={sym}" / f"year={y}" / f"month={m:02d}"
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / f"date={d}.parquet"
        part = part[["symbol", "open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
        part.to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
        print(f"[write] {sym} {d} rows={len(part)} -> {fn}")
        total += len(part)
    return total

def _stocks_write_monthly_files(df: pd.DataFrame, out_root: Path) -> int:
    """
    Write one parquet per (symbol, year, month): out/1440min/symbol=.../year=YYYY/part-YYYY-MM.parquet
    """
    if df.empty:
        return 0
    df = df.copy()
    df["year"] = df["open_time"].dt.year.astype("int16")
    df["month"] = df["open_time"].dt.month.astype("int8")
    total = 0
    for (sym, y, m), part in df.groupby(["symbol", "year", "month"], sort=True):
        outdir = out_root / "1440min" / f"symbol={sym}" / f"year={y}"
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / f"part-{y}-{m:02d}.parquet"
        part = part[["symbol", "open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
        part.to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
        print(f"[write] {sym} {y}-{m:02d} rows={len(part)} -> {fn}")
        total += len(part)
    return total

def _stocks_write_yearly_files(df: pd.DataFrame, out_root: Path) -> int:
    """
    Write one parquet per (symbol, year): out/1440min/symbol=.../part-YYYY.parquet
    """
    if df.empty:
        return 0
    df = df.copy()
    df["year"] = df["open_time"].dt.year.astype("int16")
    total = 0
    for (sym, y), part in df.groupby(["symbol", "year"], sort=True):
        outdir = out_root / "1440min" / f"symbol={sym}"
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / f"part-{y}.parquet"
        part = part[["symbol", "open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
        part.to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
        print(f"[write] {sym} {y} rows={len(part)} -> {fn}")
        total += len(part)
    return total


def cmd_stocks(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stocks: Yahoo Finance daily -> Parquet (1440min partitions)")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated tickers (e.g., AAPL,MSFT or 2330,0050)")
    ap.add_argument(
        "--tw_universe",
        type=str,
        default="none",
        choices=["none", "listed", "otc", "all"],
        help="Auto-load Taiwan market symbols if --symbols is empty: listed(.TW) / otc(.TWO) / all",
    )
    ap.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, exclusive). Default: today (UTC)")
    ap.add_argument("--out", type=str, default="data_stocks_daily", help="Output root folder")
    ap.add_argument(
        "--stocks_storage",
        type=str,
        default="monthly",
        choices=["daily", "monthly", "yearly"],
        help="Storage layout for stocks. daily=one file per day; monthly=one file per month; yearly=one file per year (default monthly).",
    )
    ap.add_argument(
        "--tw_suffix",
        type=str,
        default=".TW",
        choices=[".TW", ".TWO", "none"],
        help="Auto-append suffix for numeric tickers (default .TW). Use '.TWO' for OTC, or 'none' to disable.",
    )
    ap.add_argument("--auto_resume", action="store_true", help="Start from (last existing date - 5 days) per symbol")
    ap.add_argument("--universe_filter", action="store_true", help="After fetching, filter TW-stock universe for training suitability")
    ap.add_argument("--universe_min_turnover", type=float, default=1e8, help="Min avg turnover (amount) per day (default 1e8)")
    ap.add_argument("--universe_min_price", type=float, default=10.0, help="Min avg close price (default 10)")
    ap.add_argument("--universe_min_days", type=int, default=500, help="Min available trading days (default 500)")
    ap.add_argument("--universe_dynamic_window", type=int, default=0, help="If >0, also compute dynamic universe using last N days liquidity")
    ap.add_argument("--universe_dynamic_date", type=str, default="latest", help="Dynamic universe cutoff date (YYYY-MM-DD) or 'latest'")
    ap.add_argument("--universe_save", type=str, default="universe_selected.txt", help="Output file name for selected symbols (in --out folder)")
    args = ap.parse_args(argv)

    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("yfinance is required. Please `pip install -r requirements.txt`.") from e

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(args.start).tz_localize("UTC")
    if args.end:
        end = pd.to_datetime(args.end).tz_localize("UTC")
    else:
        end = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC") + pd.Timedelta(days=1)

    def _load_tw_universe(which: str) -> list[str]:
        """
        Load Taiwan stock codes dynamically (listed/otc/all).

        Data sources (public):
        - Listed (TWSE): https://openapi.twse.com.tw/v1/opendata/t187ap03_L  (JSON)
        - OTC (TPEx)   : http://mopsfin.twse.com.tw/opendata/t187ap03_O.csv  (CSV)
        """
        which = which.lower()
        if which == "none":
            return []

        codes_tw: set[str] = set()
        codes_two: set[str] = set()

        def _keep_code(x: str) -> bool:
            x = str(x).strip()
            return re.fullmatch(r"\d{4,6}", x) is not None

        if which in ("listed", "all"):
            url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    # common keys: 公司代號 / 公司名稱 / 上市日期 ...
                    for row in data:
                        if not isinstance(row, dict):
                            continue
                        code = row.get("公司代號") or row.get("公司代號 ") or row.get("公司代碼") or row.get("code")
                        if code is None:
                            continue
                        code = str(code).strip()
                        if _keep_code(code):
                            codes_tw.add(code)
            except Exception as e:
                raise RuntimeError(f"Failed to load TWSE listed codes from {url}: {type(e).__name__}: {e}") from e

        if which in ("otc", "all"):
            # NOTE: some networks cannot resolve mopsfin.twse.com.tw (DNS). Provide fallbacks.
            otc_sources = [
                "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv",
                "http://mopsfin.twse.com.tw/opendata/t187ap03_O.csv",
                # community-maintained fallback (useful when official host is unreachable)
                "https://raw.githubusercontent.com/mlouielu/twstock/master/twstock/codes/tpex_equities.csv",
            ]

            last_err: Exception | None = None
            for url in otc_sources:
                try:
                    df_otc = pd.read_csv(url, dtype=str, encoding_errors="ignore")
                    col = "公司代號" if "公司代號" in df_otc.columns else ("code" if "code" in df_otc.columns else df_otc.columns[0])
                    for code in df_otc[col].dropna().astype(str).map(str.strip).tolist():
                        if _keep_code(code):
                            codes_two.add(code)
                    if codes_two:
                        print(f"[tw_universe] OTC codes loaded: {len(codes_two)} from {url}")
                        last_err = None
                        break
                except Exception as e:
                    last_err = e
                    continue

            if not codes_two:
                # degrade gracefully: still allow "all" to proceed with listed only
                msg = f"[tw_universe] WARN: failed to load OTC codes from all sources; proceeding without .TWO"
                if last_err is not None:
                    msg += f" (last_error={type(last_err).__name__}: {last_err})"
                print(msg)

        out: list[str] = []
        out.extend(sorted(f"{c}.TW" for c in codes_tw))
        out.extend(sorted(f"{c}.TWO" for c in codes_two))
        return out

    raw_symbols = _norm_symbols(args.symbols) if args.symbols else []
    if not raw_symbols and args.tw_universe != "none":
        syms = _load_tw_universe(args.tw_universe)
        # cache a copy for reproducibility / quick inspection
        cache_path = out_root / f"tw_universe_{args.tw_universe}.txt"
        cache_path.write_text("\n".join(syms) + ("\n" if syms else ""), encoding="utf-8")
        print(f"[tw_universe] loaded {len(syms)} symbols ({args.tw_universe}) -> {cache_path}")
    else:
        syms = [_apply_tw_suffix(s, args.tw_suffix) for s in raw_symbols]

    print(f"[info] range={start.date()}..{end.date()} (end exclusive) out={out_root}")
    print(f"[info] symbols={','.join(syms)} tw_suffix={args.tw_suffix} auto_resume={args.auto_resume}")
    print(f"[info] stocks_storage={args.stocks_storage}")

    all_daily: list[pd.DataFrame] = []

    for sym in syms:
        sym_start = start
        if args.auto_resume:
            last_ts = _stocks_find_last_open_time(out_root, sym, args.stocks_storage)
            if last_ts is not None and pd.notna(last_ts):
                sym_start = pd.to_datetime(last_ts).tz_convert("UTC") - pd.Timedelta(days=5)
                if sym_start < start:
                    sym_start = start
                print(f"[resume] {sym}: last open_time={last_ts} -> start from {sym_start.date()}")
            else:
                print(f"[resume] {sym}: no existing files -> start from {sym_start.date()}")

        print(f"[fetch] {sym} {sym_start.date()}..{end.date()} ...")
        tkr = yf.Ticker(sym)
        hist = tkr.history(start=sym_start.date().isoformat(), end=end.date().isoformat(), interval="1d", auto_adjust=False)
        df = _stocks_to_daily_bars(sym, hist)
        if df.empty:
            print(f"[skip] {sym}: empty history")
            continue
        all_daily.append(df)
        if args.stocks_storage == "daily":
            rows = _stocks_write_daily_partitions(df, out_root)
        elif args.stocks_storage == "yearly":
            rows = _stocks_write_yearly_files(df, out_root)
        else:
            rows = _stocks_write_monthly_files(df, out_root)
        print(f"[done] {sym}: rows={rows}")

    # ---- Universe filter (TW training suitability) ----
    def _filter_universe(
        df: pd.DataFrame,
        min_turnover: float = 1e8,
        min_price: float = 10.0,
        min_days: int = 500,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        df 必須包含：
        ['symbol', 'date', 'close', 'volume', 'amount']
        """
        df = df.dropna(subset=["close", "volume", "amount"])

        stats = (
            df.groupby("symbol")
            .agg({"date": "count", "amount": "mean", "close": "mean"})
            .rename(columns={"date": "n_days", "amount": "avg_turnover", "close": "avg_price"})
        )

        cond = (stats["n_days"] >= min_days) & (stats["avg_turnover"] >= min_turnover) & (stats["avg_price"] >= min_price)
        selected_symbols = stats[cond].index.tolist()
        print(f"[Universe] total={len(stats)} → selected={len(selected_symbols)}")
        return df[df["symbol"].isin(selected_symbols)].copy(), selected_symbols

    def _filter_universe_by_date(df: pd.DataFrame, date: str, window: int = 60, min_turnover: float = 1e8) -> list[str]:
        """
        用最近 window 天的流動性來篩（以 amount 平均值為準）
        """
        sub = df[df["date"] <= date].sort_values("date").groupby("symbol").tail(window)
        stats = sub.groupby("symbol").agg({"amount": "mean"})
        selected = stats[stats["amount"] > min_turnover].index.tolist()
        return selected

    if args.universe_filter:
        if not all_daily:
            print("[Universe] No fetched rows; skip universe filtering.")
            return 0

        u = pd.concat(all_daily, ignore_index=True)
        # Align with your expected columns
        u["date"] = (u["open_time"] - pd.Timedelta(hours=8)).dt.floor("D").dt.date.astype(str)
        u["amount"] = (u["close"].astype(float) * u["volume"].astype(float)).astype(float)

        _, selected = _filter_universe(
            u[["symbol", "date", "close", "volume", "amount"]],
            min_turnover=float(args.universe_min_turnover),
            min_price=float(args.universe_min_price),
            min_days=int(args.universe_min_days),
        )

        save_path = out_root / str(args.universe_save)
        save_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
        print(f"[Universe] saved selected symbols -> {save_path}")

        if args.universe_dynamic_window and args.universe_dynamic_window > 0:
            if args.universe_dynamic_date == "latest":
                cutoff = u["date"].max()
            else:
                cutoff = args.universe_dynamic_date
            dyn = _filter_universe_by_date(
                u[["symbol", "date", "amount"]],
                date=str(cutoff),
                window=int(args.universe_dynamic_window),
                min_turnover=float(args.universe_min_turnover),
            )
            dyn_path = out_root / f"universe_dynamic_{cutoff}_w{int(args.universe_dynamic_window)}.txt"
            dyn_path.write_text("\n".join(dyn) + ("\n" if dyn else ""), encoding="utf-8")
            print(f"[Universe] dynamic({cutoff}, window={args.universe_dynamic_window}) selected={len(dyn)} -> {dyn_path}")
    return 0


def main() -> int:
    top = argparse.ArgumentParser(description="Fetch market data to Parquet (crypto / stocks)")
    sub = top.add_subparsers(dest="market", required=True)

    # keep help short; detailed args live in cmd_* parsers
    sub.add_parser("crypto", help="Binance spot 1m klines (Taipei-day parquet files)")
    sub.add_parser("stocks", help="Yahoo Finance daily bars via yfinance (US + TW)")

    ns, rest = top.parse_known_args()
    if ns.market == "crypto":
        return cmd_crypto(rest)
    if ns.market == "stocks":
        return cmd_stocks(rest)
    raise SystemExit(f"Unknown market: {ns.market}")


if __name__ == "__main__":
    raise SystemExit(main())

