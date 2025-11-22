
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming daily downloader with auto-resume:
  - One Parquet per Taipei day (08:00+8 → next 07:59+8)
  - If --auto_resume is set, for each symbol we scan the output dir and start from
    (last existing date - 1 day). If no existing data, use --start.
"""
import re
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

BINANCE_API = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
LIMIT = 1000
TPE = timezone(timedelta(hours=8))

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_range(symbol: str, start: datetime, end: datetime, session: requests.Session, retry=5, cooldown=1.0):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(minutes=LIMIT), end)
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": to_ms(cur),
            "endTime": to_ms(nxt) - 1,
            "limit": LIMIT,
        }
        for attempt in range(retry):
            try:
                r = session.get(BINANCE_API, params=params, timeout=30)
                if r.status_code in (418, 451):
                    raise RuntimeError(f"HTTP {r.status_code} (IP banned/geo blocked).")
                if r.status_code == 429:
                    sleep_t = cooldown * (2 ** attempt)
                    print(f"[429] throttled; sleeping {sleep_t:.1f}s")
                    time.sleep(sleep_t)
                    continue
                r.raise_for_status()
                data = r.json()
                if not data:
                    print(f"[empty] {symbol} {cur:%Y-%m-%d %H:%M}..{nxt:%Y-%m-%d %H:%M}")
                    break
                cols = [
                    "open_time","open","high","low","close","volume",
                    "close_time","quote_asset_volume","number_of_trades",
                    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
                ]
                df = pd.DataFrame(data, columns=cols)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                num_cols = ["open","high","low","close","volume","quote_asset_volume",
                            "taker_buy_base_volume","taker_buy_quote_volume"]
                df[num_cols] = df[num_cols].astype(float)
                df["number_of_trades"] = df["number_of_trades"].astype("int32")
                df["symbol"] = symbol
                print(f"[fetch] {symbol} {cur:%Y-%m-%d %H:%M}..{nxt:%Y-%m-%d %H:%M} -> {len(df)} rows")
                yield df[["symbol","open_time","open","high","low","close","volume",
                          "close_time","quote_asset_volume","number_of_trades",
                          "taker_buy_base_volume","taker_buy_quote_volume"]]
                break
            except Exception as e:
                if attempt == retry - 1:
                    raise
                sleep_t = cooldown * (2 ** attempt)
                print(f"[error] {type(e).__name__}: {e}; retry in {sleep_t:.1f}s")
                time.sleep(sleep_t)
        cur = nxt

def write_day(symbol: str, day_tpe: datetime, out_root: Path):
    start_tpe = day_tpe.replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=TPE)
    end_tpe = (start_tpe + timedelta(days=1))
    start_utc = start_tpe.astimezone(timezone.utc)
    end_utc = end_tpe.astimezone(timezone.utc)

    parts = []
    for df in fetch_range(symbol, start_utc, end_utc, requests.Session()):
        parts.append(df)
    if not parts:
        print(f"[skip] {symbol} {day_tpe.date()} (no data)")
        return 0
    big = pd.concat(parts, ignore_index=True).sort_values("open_time")
    big = big[(big["open_time"] >= start_utc) & (big["open_time"] < end_utc)].reset_index(drop=True)

    y, m = day_tpe.year, day_tpe.month
    outdir = out_root / f"symbol={symbol}"/ f"year={y}"/ f"month={m:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"date={day_tpe.date()}.parquet"
    big.to_parquet(fn, engine="pyarrow", compression="zstd", index=False)
    print(f"[write] {symbol} {day_tpe.date()} rows={len(big)} -> {fn}")
    return len(big)

def daterange_days_tpe(start: datetime, end: datetime):
    start_tpe = start.astimezone(TPE).date()
    end_tpe = end.astimezone(TPE).date()
    cur = datetime.combine(start_tpe, datetime.min.time()).replace(tzinfo=TPE)
    while cur.date() < end_tpe:
        yield cur
        cur += timedelta(days=1)

def find_last_tpe_date(out_root: Path, symbol: str):
    pattern = re.compile(r"date=(\d{4}-\d{2}-\d{2})\.parquet$")
    base = out_root / f"symbol={symbol}"
    if not base.exists():
        return None
    latest = None
    for p in base.rglob("date=*.parquet"):
        m = pattern.search(str(p))
        if not m:
            continue
        d = m.group(1)
        if (latest is None) or (d > latest):
            latest = d
    return latest

def main():
    ap = argparse.ArgumentParser(description="Streaming daily 1m klines (Binance) with auto-resume")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT")
    ap.add_argument("--start", type=str, default="2020-08-21", help="UTC date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default="2025-08-21", help="UTC date (YYYY-MM-DD, exclusive)")
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--auto_resume", action="store_true", help="Start from (last existing TPE date - 1 day) per symbol")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    user_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    out_root = Path(args.out)

    print(f"End (UTC): {end} (exclusive) | auto_resume={args.auto_resume}")
    for sym in symbols:
        start = user_start
        if args.auto_resume:
            last = find_last_tpe_date(out_root, sym)
            if last:
                last_dt = datetime.fromisoformat(last).replace(tzinfo=TPE)
                resume_dt = (last_dt - timedelta(days=1))
                start = resume_dt.astimezone(timezone.utc)
                print(f"[resume] {sym}: found last TPE date={last} -> resume from previous day {resume_dt.date()} (UTC start {start.date()})")
            else:
                print(f"[resume] {sym}: no existing files, use user --start {start.date()}")

        print(f"=== {sym} === Date range start={start} end={end}")
        total = 0
        for day_tpe in daterange_days_tpe(start, end):
            total += write_day(sym, day_tpe, out_root)
        print(f"[done] {sym} total rows={total}")

if __name__ == "__main__":
    main()

# 從上次爬取的日期接續爬取資料
# python fetch_klines_parquet.py --symbols [要爬取的幣種A,要爬取的幣種B,...] --end [爬取結束日期的後一天] --out [參考最後日期與輸出的資料夾] --auto_resume
# Ex: python fetch_klines_parquet.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,XRPUSDT --end 2025-10-27 --out .\data --auto_resume
# 日期設為 2025-10-27，則爬取到 2025-10-26 為止
