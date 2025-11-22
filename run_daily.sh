
#!/usr/bin/env bash
# Daily job: auto-resume fetch (Taipei day) + build features (daily layout)
# Usage:
#   ./run_daily.sh "BTCUSDT,ETHUSDT,BNBUSDT" /data/data_daily /data/features_daily
# All args optional.
set -euo pipefail

SYMBOLS="${1:-BTCUSDT,ETHUSDT,BNBUSDT}"
OUT_DIR="${2:-data_daily}"
FEAT_DIR="${3:-features_daily}"

# End date is today (UTC), exclusive
END=$(date -u +%F)

echo "[fetch] symbols=${SYMBOLS} out=${OUT_DIR} end=${END} (auto-resume per symbol)"
python fetch_klines_parquet_daily_stream.py --symbols "${SYMBOLS}" --out "${OUT_DIR}" --end "${END}" --auto_resume

echo "[features] src=${OUT_DIR} -> dst=${FEAT_DIR}"
python feature_builder.py --src "${OUT_DIR}" --dst "${FEAT_DIR}" --symbols "${SYMBOLS}" --start 2000-01-01 --end "${END}"

echo "Done."
