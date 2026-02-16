# 🚀 Transformer Preprocessing 快速參考卡

## 核心 3 個函數

### 1️⃣ 去冗餘
```python
from train_transform import reduce_features_transformer

features_reduced = reduce_features_transformer(features_by_category)
# ~50 特徵 → ~30 特徵
```

**規則速查表**：
| 特徵類型 | 行為 |
|---------|-----|
| sma_* | ❌ **全刪** |
| ema_* | ✅ 只保留 slope & dev |
| bb_* | ✅ 只保留 width & pos |
| don_* | ✅ **都保留** |
| rv_* | ✅ 只保留 60, 240, pctile |
| vol_z_*, trades_z_* | ✅ **都保留** |
| hour_*, dow_* | ✅ **都保留** |

---

### 2️⃣ 正規化（rolling z-score）
```python
from train_transform import normalize_features_transformer

df_normalized = normalize_features_transformer(
    df_features, 
    feature_list, 
    window=500  # 500 個 bar（~3-5 天 @ 10min）
)
```

**規則速查表**：
| 特徵類型 | 正規化 | 理由 |
|---------|------|------|
| ret_* | ❌ 不 | 保留 fat tail |
| ema_*, bb_*, don_* | ✅ rolling z | 防止尺度漂移 |
| rv_*, atr_*, park_* | ✅ rolling z | 正規化波動 |
| vol_z_*, cmf_*, mfi_* | ✅ rolling z | 穩定訓練 |
| hour_*, dow_* | ❌ 不 | 已 encoded |

---

### 3️⃣ 分離 Label（**很重要！**）
```python
from train_transform import drop_labels_from_features

df_X = drop_labels_from_features(df_all)  # Feature only
df_y = df_all[[c for c in df_all.columns if c.startswith('y_')]]  # Label only
```

---

## 完整 Workflow（Copy-Paste）

```python
import pandas as pd
from train_transform import (
    infer_feature_columns,
    reduce_features_transformer,
    drop_labels_from_features,
    normalize_features_transformer
)

# 假設 df_all 已載入
df_all = pd.read_parquet('merged_data.parquet')

# Step 1: 分類
features_by_cat, feats_all = infer_feature_columns(df_all, exclude=[])

# Step 2: 去冗餘
feats_reduced = reduce_features_transformer(features_by_cat)
print(f"Features: {len(feats_all)} → {len(feats_reduced)}")

# Step 3: 分離
df_X = drop_labels_from_features(df_all)

# Step 4: 正規化
df_X = normalize_features_transformer(df_X, feats_reduced, window=500)

# Step 5: 分開提取
X = df_X[feats_reduced].values      # shape: (n, d)
y = df_all[[c for c in df_all.columns if c.startswith('y_')]].values  # shape: (n, k)

# 現在可以直接餵進 Transformer
model.fit(X, y)
```

---

## 預期結果

```
[step 2] Feature Reduction (去冗餘)
Before: 48 features → After: 31 features ✓
Removed: 17 redundant features

[step 5] Final Feature Statistics
✓ Feature matrix shape: (95872, 31)
✓ Feature columns: 31
✓ Label matrix shape: (95872, 6)
✓ Label columns: 6
  - y_cls_sign_60m
  - y_reg_ret_60m
  - y_tp_sl_60m
  - y_tb_60m
  - y_vol_60m
  - y_tp_sl_60m_t_hit
```

---

## 常見問題 Q&A

### Q1: 為什麼要刪除 SMA？
**A**: 
- SMA 與 EMA 高度共線（相關係數 >0.95）
- Transformer 的 attention 機制會自己學習時序模式
- 冗餘特徵只會增加訓練成本，不會提升性能

### Q2: rolling window=500 為什麼?
**A**:
- 10 分鐘 K 線 × 500 = 3,333 分鐘 ≈ 2.3 天
- 適合短期 regime 變化
- 避免過度利用古老歷史資料

### Q3: 為什麼 ret_* 不正規化?
**A**:
- 報酬天然是 scale-free（已經是對數差分）
- Fat tail 對金融很重要（extreme events 信息豐富）
- 全局 z-score 會破壞 fat tail 結構

### Q4: 時間特徵用 sin/cos 還是 one-hot?
**A**:
- sin/cos（已在 feature_builder.py 實現）✅
- 原因：
  - 保留周期性（hour 23 和 hour 0 相近）
  - 更緊湊（2 維 vs 24 維）
  - Transformer 更容易學

### Q5: 要預先移除 Label 列嗎？
**A**:
- **必須！** ✅ 用 `drop_labels_from_features()`
- 原因：
  - 防止 model 直接學習 target（information leak）
  - 特別是 `_t_hit` 列會洩露時間
  - 嚴格區分 X 和 y

---

## 🎯 最小化可行代碼

**如果只想快速測試**：

```python
# 最簡版本（3 行核心邏輯）
from train_transform import reduce_features_transformer, drop_labels_from_features, normalize_features_transformer

df = pd.read_parquet('data.parquet')
feats = reduce_features_transformer({...})  # 去冗餘
df = drop_labels_from_features(df)          # 分離 Label
df = normalize_features_transformer(df, feats, window=500)  # 正規化

# 交給 Transformer
model.fit(df[feats], y)
```

---

## ⏱️ 執行時間

| 步驟 | 數據量 | 時間 |
|-----|------|------|
| infer_feature_columns | 100K rows | < 1s |
| reduce_features | - | < 1s |
| drop_labels | - | < 1s |
| rolling_normalize | 100K rows | ~5-10s |
| **總計** | | ~10-15s |

---

## 🚨 必檢清單

在訓練前確認：

- [ ] 所有 Label 列都不在 feature matrix 中（用 `print(df_X.columns)` 檢查）
- [ ] 時間序列切割（not random split）
- [ ] rolling window 大小合理（相對於 K 線周期）
- [ ] 沒有 NaN（檢查 `df_X.isna().sum()`）
- [ ] Feature 數量合理（~30-40 個）
- [ ] Label 形狀正確（(n_samples, n_label_columns)）

---

## 🔗 相關檔案

```
train_transform.py              ← 主腳本（含新函數）
prep_transformer_data.py        ← 獨立 pipeline
TRANSFORMER_PREPROCESSING_GUIDE.md  ← 詳細文檔
```

---

**版本**: 1.0 | **日期**: 2026-01-05 | **作者**: Transformer Preprocessing Refactor
