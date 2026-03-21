
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

## 實驗結果
### 20260322
目前獲得最高 IC 的結果:
```
2026-03-21 22:36:47 INFO
【預處理完成】
2026-03-21 22:36:47 INFO   • 原始特徵: 86 → 減少後: 59
2026-03-21 22:36:47 INFO   • 移除冗餘: 27 個特徵
2026-03-21 22:36:47 INFO   • Label 行數: 12452
2026-03-21 22:36:47 INFO   • 時間範圍: 2018-01-01 00:00:00+00:00 → 2025-08-22 00:00:00+00:00
[info] 使用 LazySeqDataset (動態產生序列，省記憶體)
[data] cls_classes=3 | tp_classes=0
2026-03-21 22:37:44 INFO [epoch 1] train=1.676416 val=1.306563 best=inf@-1
2026-03-21 22:38:18 INFO [epoch 2] train=1.673914 val=1.304480 best=1.306563@1
2026-03-21 22:38:51 INFO [epoch 3] train=1.671457 val=1.303396 best=1.304480@2
2026-03-21 22:39:25 INFO [epoch 4] train=1.669379 val=1.304239 best=1.303396@3
2026-03-21 22:39:58 INFO [epoch 5] train=1.664845 val=1.303007 best=1.303396@3
2026-03-21 22:40:32 INFO [epoch 6] train=1.673643 val=1.302591 best=1.303007@5
2026-03-21 22:41:05 INFO [epoch 7] train=1.665931 val=1.303240 best=1.302591@6
2026-03-21 22:41:39 INFO [epoch 8] train=1.667164 val=1.303751 best=1.302591@6
2026-03-21 22:42:13 INFO [epoch 9] train=1.670685 val=1.302661 best=1.302591@6
2026-03-21 22:42:47 INFO [epoch 10] train=1.664509 val=1.302345 best=1.302591@6
2026-03-21 22:43:20 INFO [epoch 11] train=1.670322 val=1.304517 best=1.302345@10
2026-03-21 22:43:53 INFO [epoch 12] train=1.669037 val=1.302427 best=1.302345@10
2026-03-21 22:44:25 INFO [epoch 13] train=1.662934 val=1.302076 best=1.302345@10
2026-03-21 22:44:59 INFO [epoch 14] train=1.662176 val=1.302507 best=1.302076@13
2026-03-21 22:45:32 INFO [epoch 15] train=1.661700 val=1.304790 best=1.302076@13
2026-03-21 22:46:03 INFO [epoch 16] train=1.667317 val=1.302570 best=1.302076@13
2026-03-21 22:46:35 INFO [epoch 17] train=1.661908 val=1.300301 best=1.302076@13
2026-03-21 22:47:07 INFO [epoch 18] train=1.661837 val=1.302874 best=1.300301@17
2026-03-21 22:47:38 INFO [epoch 19] train=1.672074 val=1.301334 best=1.300301@17
2026-03-21 22:48:10 INFO [epoch 20] train=1.664979 val=1.302931 best=1.300301@17
2026-03-21 22:48:42 INFO [epoch 21] train=1.661637 val=1.301743 best=1.300301@17
2026-03-21 22:49:14 INFO [epoch 22] train=1.658171 val=1.297993 best=1.300301@17
2026-03-21 22:49:47 INFO [epoch 23] train=1.663208 val=1.299748 best=1.297993@22
2026-03-21 22:50:19 INFO [epoch 24] train=1.667372 val=1.298253 best=1.297993@22
2026-03-21 22:50:51 INFO [epoch 25] train=1.664253 val=1.296702 best=1.297993@22
2026-03-21 22:51:24 INFO [epoch 26] train=1.665101 val=1.298661 best=1.296702@25
2026-03-21 22:51:54 INFO [epoch 27] train=1.660611 val=1.297178 best=1.296702@25
2026-03-21 22:52:25 INFO [epoch 28] train=1.656070 val=1.301188 best=1.296702@25
2026-03-21 22:52:58 INFO [epoch 29] train=1.651850 val=1.297080 best=1.296702@25
2026-03-21 22:53:30 INFO [epoch 30] train=1.664124 val=1.295320 best=1.296702@25
2026-03-21 22:54:02 INFO [epoch 31] train=1.667429 val=1.297921 best=1.295320@30
2026-03-21 22:54:34 INFO [epoch 32] train=1.657200 val=1.302291 best=1.295320@30
2026-03-21 22:55:05 INFO [epoch 33] train=1.663948 val=1.297041 best=1.295320@30
2026-03-21 22:55:38 INFO [epoch 34] train=1.665156 val=1.297186 best=1.295320@30
2026-03-21 22:56:09 INFO [epoch 35] train=1.661400 val=1.302632 best=1.295320@30
2026-03-21 22:56:41 INFO [epoch 36] train=1.658384 val=1.298455 best=1.295320@30
2026-03-21 22:57:13 INFO [epoch 37] train=1.665250 val=1.301094 best=1.295320@30
2026-03-21 22:57:45 INFO [epoch 38] train=1.652878 val=1.302814 best=1.295320@30
2026-03-21 22:58:16 INFO [epoch 39] train=1.641967 val=1.301240 best=1.295320@30
2026-03-21 22:58:47 INFO [epoch 40] train=1.648968 val=1.297931 best=1.295320@30
2026-03-21 22:59:19 INFO [epoch 41] train=1.647980 val=1.319431 best=1.295320@30
2026-03-21 22:59:49 INFO [epoch 42] train=1.654850 val=1.297775 best=1.295320@30
2026-03-21 23:00:21 INFO [epoch 43] train=1.644353 val=1.304138 best=1.295320@30
2026-03-21 23:00:53 INFO [epoch 44] train=1.652773 val=1.303459 best=1.295320@30
2026-03-21 23:01:25 INFO [epoch 45] train=1.649472 val=1.297871 best=1.295320@30
2026-03-21 23:01:56 INFO [epoch 46] train=1.649082 val=1.301036 best=1.295320@30
2026-03-21 23:02:28 INFO [epoch 47] train=1.646113 val=1.294884 best=1.295320@30
2026-03-21 23:02:59 INFO [epoch 48] train=1.643237 val=1.332697 best=1.294884@47
2026-03-21 23:03:31 INFO [epoch 49] train=1.643482 val=1.314749 best=1.294884@47
2026-03-21 23:04:03 INFO [epoch 50] train=1.639600 val=1.310267 best=1.294884@47
2026-03-21 23:04:34 INFO [epoch 51] train=1.643175 val=1.366077 best=1.294884@47
2026-03-21 23:05:06 INFO [epoch 52] train=1.644789 val=1.311153 best=1.294884@47
2026-03-21 23:05:37 INFO [epoch 53] train=1.637681 val=1.303000 best=1.294884@47
2026-03-21 23:06:10 INFO [epoch 54] train=1.632248 val=1.340982 best=1.294884@47
2026-03-21 23:06:41 INFO [epoch 55] train=1.639700 val=1.310425 best=1.294884@47
2026-03-21 23:07:14 INFO [epoch 56] train=1.637222 val=1.315019 best=1.294884@47
2026-03-21 23:07:46 INFO [epoch 57] train=1.633820 val=1.324138 best=1.294884@47
2026-03-21 23:08:23 INFO [epoch 58] train=1.622544 val=1.323972 best=1.294884@47
2026-03-21 23:08:56 INFO [epoch 59] train=1.625040 val=1.340009 best=1.294884@47
2026-03-21 23:09:28 INFO [epoch 60] train=1.621682 val=1.304991 best=1.294884@47
2026-03-21 23:10:00 INFO [epoch 61] train=1.620415 val=1.315836 best=1.294884@47
2026-03-21 23:10:32 INFO [epoch 62] train=1.625396 val=1.311304 best=1.294884@47
2026-03-21 23:11:04 INFO [epoch 63] train=1.628294 val=1.321255 best=1.294884@47
2026-03-21 23:11:35 INFO [epoch 64] train=1.614520 val=1.326767 best=1.294884@47
2026-03-21 23:12:07 INFO [epoch 65] train=1.611210 val=1.376871 best=1.294884@47
2026-03-21 23:12:39 INFO [epoch 66] train=1.614113 val=1.386812 best=1.294884@47
2026-03-21 23:13:11 INFO [epoch 67] train=1.602977 val=1.316522 best=1.294884@47
2026-03-21 23:13:43 INFO [epoch 68] train=1.618387 val=1.308477 best=1.294884@47
2026-03-21 23:14:15 INFO [epoch 69] train=1.608826 val=1.399667 best=1.294884@47
2026-03-21 23:14:46 INFO [epoch 70] train=1.606784 val=1.326862 best=1.294884@47
2026-03-21 23:15:19 INFO [epoch 71] train=1.606776 val=1.306322 best=1.294884@47
2026-03-21 23:15:50 INFO [epoch 72] train=1.608374 val=1.405889 best=1.294884@47
2026-03-21 23:16:22 INFO [epoch 73] train=1.604035 val=1.330847 best=1.294884@47
2026-03-21 23:16:52 INFO [epoch 74] train=1.597262 val=1.359247 best=1.294884@47
2026-03-21 23:17:22 INFO [epoch 75] train=1.594307 val=1.341494 best=1.294884@47
2026-03-21 23:17:55 INFO [epoch 76] train=1.581815 val=1.344065 best=1.294884@47
2026-03-21 23:18:27 INFO [epoch 77] train=1.598960 val=1.370282 best=1.294884@47
2026-03-21 23:19:00 INFO [epoch 78] train=1.590874 val=1.413771 best=1.294884@47
2026-03-21 23:19:33 INFO [epoch 79] train=1.582795 val=1.386289 best=1.294884@47
2026-03-21 23:20:06 INFO [epoch 80] train=1.584017 val=1.405515 best=1.294884@47
2026-03-21 23:20:38 INFO [epoch 81] train=1.572427 val=1.391463 best=1.294884@47
2026-03-21 23:21:10 INFO [epoch 82] train=1.572155 val=1.406108 best=1.294884@47
2026-03-21 23:21:43 INFO [epoch 83] train=1.569579 val=1.488643 best=1.294884@47
2026-03-21 23:22:15 INFO [epoch 84] train=1.563240 val=1.421368 best=1.294884@47
2026-03-21 23:22:47 INFO [epoch 85] train=1.576672 val=1.362240 best=1.294884@47
2026-03-21 23:23:18 INFO [epoch 86] train=1.577174 val=1.378293 best=1.294884@47
2026-03-21 23:23:48 INFO [epoch 87] train=1.571283 val=1.363497 best=1.294884@47
2026-03-21 23:23:48 INFO [early-stop] best=1.294884 @ epoch 47
2026-03-21 23:23:57 INFO [test] {'acc': 0.4364232977850697, 'mae': 0.025642098512094074, 'ic': 0.12106571995187487}
2026-03-21 23:23:57 INFO [test] {'acc': 0.4364232977850697, 'mae': 0.025642098512094074, 'ic': 0.12106571995187487}
2026-03-21 23:23:57 INFO [done] saved artefacts to logs\train_20260321_223636_BTCUSDT_ETHUSDT_BNBUSDT_DOGEUSDT_SOLUSDT
2026-03-21 23:23:57 INFO 訓練完成，metrics: {'acc': 0.4364232977850697, 'mae': 0.025642098512094074, 'ic': 0.12106571995187487}    
2026-03-21 23:23:57 INFO === Transformer Training End ===
2026-03-21 23:23:57 INFO Log dir: logs\train_20260321_223636_BTCUSDT_ETHUSDT_BNBUSDT_DOGEUSDT_SOLUSDT
```
針對 `feature_builder.py`、`label_builder.py`、`train_transformer.py` 的指令:
* `python feature_builder.py --src data --dst features --symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,XRPUSDT,AVAXUSDT,MATICUSDT,LINKUSDT,DOTUSDT,TRXUSDT,UNIUSDT,ATOMUSDT --start 2018-01-01 --end 2026-03-20 --horizon_m 1440 --interval 1440 --enforce_continuous`
* `label_builder.py --src data --dst labels --symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,XRPUSDT,AVAXUSDT,MATICUSDT,LINKUSDT,DOTUSDT,TRXUSDT,UNIUSDT,ATOMUSDT --start 2018-01-01 --end 2026-03-20 --horizon_m 1440 --interval 1440 --enforce_continuous`
* `python train_transformer.py --features_root features --labels_root labels --symbols BTCUSDT,ETHUSDT,BNBUSDT,DOGEUSDT,SOLUSDT --start 2018-01-01 --end 2025-08-23 --seq_len 7 --layers 5 --hidden 256 --n_heads 8 --dropout 0.2 --epochs 100 --patience 40 --lr 1e-3 --weight_decay 1e-4 --batch_size 512 --interval 1440 --weight_mode none --label_cls y_cls_sign_1440m --label_reg y_reg_ret_1440m`

---
祝你建模順利！🚀
