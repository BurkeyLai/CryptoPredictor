
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED (kept for backward compatibility)
------------------------------------------
Crypto downloader has been merged into `fetch_market_parquet.py`.

New usage:
  python fetch_market_parquet.py crypto --symbols BTCUSDT,ETHUSDT --out data --auto_resume
"""

from __future__ import annotations

import sys

from fetch_market_parquet import cmd_crypto


def main() -> int:
    # keep the old CLI shape by forwarding all args to the crypto subcommand
    return cmd_crypto(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

# 從上次爬取的日期接續爬取資料
# python fetch_klines_parquet.py --symbols [要爬取的幣種A,要爬取的幣種B,...] --end [爬取結束日期的後一天] --out [參考最後日期與輸出的資料夾] --auto_resume
# Ex: python fetch_klines_parquet.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,XRPUSDT --end 2025-10-27 --out .\data --auto_resume
# 日期設為 2025-10-27，則爬取到 2025-10-26 為止
# python fetch_klines_parquet.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,XRPUSDT,AVAXUSDT,MATICUSDT,LINKUSDT,DOTUSDT,TRXUSDT,UNIUSDT,ATOMUSDT --end 2026-03-20 --out .\data --auto_resume
