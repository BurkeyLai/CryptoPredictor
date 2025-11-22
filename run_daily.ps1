
# PowerShell Daily job: auto-resume fetch (Taipei day) + build features (daily layout)
# Usage:
#   .\run_daily.ps1 -Symbols "BTCUSDT,ETHUSDT,BNBUSDT" -OutDir "data_daily" -FeatDir "features_daily"
param(
  [string]$Symbols = "BTCUSDT,ETHUSDT,BNBUSDT",
  [string]$OutDir = "data_daily",
  [string]$FeatDir = "features_daily"
)

$end = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")  # exclusive

Write-Output "[fetch] symbols=$Symbols out=$OutDir end=$end (auto-resume per symbol)"
python fetch_klines_parquet_daily_stream.py --symbols $Symbols --out $OutDir --end $end --auto_resume

Write-Output "[features] src=$OutDir -> dst=$FeatDir"
python feature_builder.py --src $OutDir --dst $FeatDir --symbols $Symbols --start 2000-01-01 --end $end

Write-Output "Done."
