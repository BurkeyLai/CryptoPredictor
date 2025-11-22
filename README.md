
# Crypto 1m K-line Starter Kit (Binance → Parquet → Features → ClickHouse)

這個工具包協助你：
1) 抓取 **Binance 現貨** 的 **1 分鐘 K 線（OHLCV）**
2) 以 **Parquet + 分區** 落地
3) 產生常用 **技術指標特徵**（MA、RSI、ATR、Realized Vol、量能異常 z-score）
4) （可選）匯入 **ClickHouse** 查詢/回測

## 專案內容
- `fetch_klines_parquet.py`：抓取 1m K 線並落地 Parquet
- `feature_builder.py`：讀取 Parquet，產生技術指標特徵（亦以 Parquet 落地）
- `clickhouse.sql`：ClickHouse 建表＋範例查詢＋從 Parquet 匯入
- `run_daily.sh` / `run_daily.ps1`：每日排程樣板（Linux/macOS / Windows）
- `cron_example.txt`：cron 例子
- `requirements.txt`：所需 Python 套件

## 安裝
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 抓歷史（預設：5 年，BTC/ETH/BNB）
```bash
python fetch_klines_parquet.py
```

## 產生特徵（覆蓋/補寫同分區檔案）
```bash
python feature_builder.py --src data --dst features --symbols BTCUSDT,ETHUSDT,BNBUSDT       --start 2020-08-21 --end 2025-08-22
```

## 每日排程
- Linux/macOS：
  ```bash
  chmod +x run_daily.sh
  ./run_daily.sh BTCUSDT,ETHUSDT,BNBUSDT 2
  ```
  編輯 `crontab -e`，加入（UTC 00:10 執行）：
  ```
  10 0 * * * cd /path/to/crypto_1m_kit && bash run_daily.sh BTCUSDT,ETHUSDT,BNBUSDT 2 >> logs/cron.log 2>&1
  ```

- Windows（PowerShell）：
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  .\run_daily.ps1 -Symbols "BTCUSDT,ETHUSDT,BNBUSDT" -DaysBack 2
  ```
  Task Scheduler 新增基本工作 → 觸發時間每日 → 動作：
  `powershell.exe -File "C:\path\to\run_daily.ps1" -Symbols "BTCUSDT,ETHUSDT,BNBUSDT" -DaysBack 2`

## ClickHouse（可選）
1. 安裝 ClickHouse（server + client）  
2. 修改 `clickhouse.sql` 中的檔案路徑，然後：
   ```bash
   clickhouse-client --multiquery < clickhouse.sql
   ```
   或手動執行其中每段 SQL。
3. 常見查詢：
   - 計算 5 分鐘 K 線
   - 查詢某段期間的 RSI/波動等特徵

## Parquet 分區結構
```
data/
  exchange=binance/
    symbol=BTCUSDT/
      year=2020/month=08/part-2020-08.parquet
      ...
features/
  exchange=binance/
    symbol=BTCUSDT/
      year=2020/month=08/features-2020-08.parquet
      ...
```

## 訓練用幣數：3 檔夠嗎？更多更好？
- **起步**：3 檔（BTC/ETH/BNB）已能做出可用模型，且資料品質高、流動性佳。
- **泛化**：建議擴到 **10–20 檔主流幣**（按成交量前列），能提升模型在不同 regime 的穩健度。
- **做法**：
  - 先以 3 檔開發與驗證流程（抓取→清洗→特徵→標籤→回測）。
  - 穩定後，再批次擴幣並重訓/微調；也可做 **跨資產合併訓練**（加入 `symbol` one-hot/embedding 或以資產內標準化的特徵）。
  - 注意 **資料洩漏**：標準化統計量要在「訓練集內 per-時間窗」計算；交叉驗證採 **時間切片**；避免跨資產洩漏（若做資產內 z-score）。

## 延伸建議
- 增加標籤產生器（ex: 預測 15/30/60 分鐘未來報酬 sign）。
- 加入交易成本與滑點模型。
- 對缺漏分鐘可開 `--enforce_continuous` 以利技術指標計算。
- 用 ClickHouse 外掛 Parquet 目錄（file()）做直接查詢，或把數據匯入表中以獲得更好的索引效能。

---
祝你建模順利！🚀
