# Transformer 架構視覺化

## 整體數據流

```
┌─────────────────────────────────────────────────────────────────┐
│                         預處理管道                              │
│  feature_builder.py → train_transform.py → data_loader.py      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    (57 features, 36 timesteps)
                             ↓
           ┌─────────────────────────────────────┐
           │   TimeSeriesTransformerModel        │
           │   ├─ TimeSeriesTransformer          │
           │   ├─ SymbolEmbedding (可選)         │
           │   ├─ Fusion Layer (可選)            │
           │   └─ MultiTaskHeads                 │
           └────────────┬────────────────────────┘
                        ↓
          ┌─────────────────────────────────┐
          │  輸出: Multi-Task Predictions   │
          ├─ cls: 分類 (3類)               │
          ├─ reg: 迴歸 (連續)              │
          ├─ vol: 波動 (連續)              │
          └─ tp: TP/SL (3維)               │
          └─────────────────────────────────┘
                        ↓
          ┌─────────────────────────────────┐
          │   訓練循環 (train_transformer)   │
          │   ├─ 前向傳播                   │
          │   ├─ 損失計算 (加權組合)       │
          │   ├─ 反向傳播                   │
          │   ├─ 優化器更新                 │
          │   └─ 提早停止                   │
          └─────────────────────────────────┘
```

## TimeSeriesTransformer 詳細架構

```
Input: (Batch=B, Seq_Len=36, Features=57)
        │
        ↓
    ┌───────────────────────────┐
    │   線性投影層              │
    │   Input Projection        │
    │   (57 → 256)              │
    └────────────┬──────────────┘
                 ↓
    ┌───────────────────────────┐
    │   位置編碼                │
    │   Positional Embedding    │
    │   (1, 36, 256)            │
    │   ~ N(0, 0.02)            │
    └────────────┬──────────────┘
                 ↓
    ┌───────────────────────────┐
    │   Dropout (p=0.1)         │
    └────────────┬──────────────┘
                 ↓
    ┌───────────────────────────────────────┐
    │   Transformer Encoder                 │
    │   ┌─────────────────────────────────┐ │
    │   │ Layer 1: Multi-Head Attention   │ │
    │   │ - 8 heads                       │ │
    │   │ - d_model=256                   │ │
    │   │ - d_head=32                     │ │
    │   │ - Self-Attention (Q,K,V)        │ │
    │   └─────────────────────────────────┘ │
    │   ↓                                   │
    │   ┌─────────────────────────────────┐ │
    │   │ Add & Norm (Residual)           │ │
    │   └─────────────────────────────────┘ │
    │   ↓                                   │
    │   ┌─────────────────────────────────┐ │
    │   │ Feed-Forward Network            │ │
    │   │ - Linear(256, 1024)             │ │
    │   │ - GELU                          │ │
    │   │ - Linear(1024, 256)             │ │
    │   └─────────────────────────────────┘ │
    │   ↓                                   │
    │   ┌─────────────────────────────────┐ │
    │   │ Add & Norm (Residual)           │ │
    │   └─────────────────────────────────┘ │
    │   (重複 4 次: num_layers=4)            │
    └────────────┬────────────────────────────┘
                 ↓
    ┌───────────────────────────┐
    │   取最後時間點            │
    │   h = x[:, -1]            │
    │   (B, 256)                │
    └────────────┬──────────────┘
                 ↓
    Output: (Batch=B, d_model=256)
```

## TimeSeriesTransformerModel 完整流程

```
┌─────────────────────────────────────────────────┐
│ 輸入 1: Time Series Data (B, 36, 57)           │
│ 輸入 2: Symbol Index (B,) [可選]               │
└─────────────┬─────────────────────────────────┘
              │
              ├─ 路徑 A: Backbone      ├─ 路徑 B: Symbol (可選)
              │                        │
              ↓                        ↓
        ┌──────────────────┐    ┌─────────────────────┐
        │ TimeSeriesXformer│    │ SymbolEmbedding     │
        │ (B, 256)         │    │ 7 → 32維            │
        │                  │    │ (B, 32)             │
        └────────┬─────────┘    └──────────┬──────────┘
                 │                         │
                 └──────────┬──────────────┘
                            ↓
                    ┌──────────────────┐
                    │ Concatenate      │
                    │ (B, 288)         │
                    │ [256+32]         │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────────────┐
                    │ Fusion Layer             │
                    │ Linear(288 → 256)        │
                    │ ReLU                     │
                    │ Dropout(0.1)             │
                    │ (B, 256)                 │
                    └────────┬─────────────────┘
                             ↓
                    ┌──────────────────┐
                    │ MultiTaskHeads   │
                    └────────┬─────────┘
                             ↓
        ┌────────────────────┬────────────────────┐
        ↓                    ↓                    ↓
    ┌────────┐         ┌────────┐         ┌────────┐
    │ cls_   │         │ reg_   │         │ vol_   │
    │ head   │         │ head   │         │ head   │
    │ (B, 3)│         │ (B,)   │         │ (B,)   │
    └────────┘         └────────┘         └────────┘
        │                  │                  │
        └──────────┬───────┴──────────┬───────┘
                   │                  │
                   ↓                  ↓
              ┌────────────────────────────┐
              │ tp_head                    │
              │ (B, 3)                     │
              └────────────────────────────┘

輸出: Dict {
    "cls": (B, 3),    # 分類 logits
    "reg": (B,),      # 迴歸值
    "vol": (B,),      # 波動值
    "tp": (B, 3)      # TP/SL 信號
}
```

## MultiTaskHeads 內部結構

### 簡單模式 (use_intermediate=False)

```
Backbone Output (B, 256)
    │
    ├─→ [Linear(256, n_cls)] → cls_logits (B, 3)
    ├─→ [Linear(256, 1)] → reg (B,)
    ├─→ [Linear(256, 1)] → vol (B,)
    └─→ [Linear(256, 3)] → tp (B, 3)
```

### 複雜模式 (use_intermediate=True)

```
Backbone Output (B, 256)
    │
    ├─→ [Linear(256→128)] [ReLU] [Dropout] [Linear(128→n_cls)]
    │   └─→ cls_logits (B, 3)
    │
    ├─→ [Linear(256→128)] [ReLU] [Dropout] [Linear(128→1)]
    │   └─→ reg (B,)
    │
    ├─→ [Linear(256→128)] [ReLU] [Dropout] [Linear(128→1)]
    │   └─→ vol (B,)
    │
    └─→ [Linear(256→128)] [ReLU] [Dropout] [Linear(128→3)]
        └─→ tp (B, 3)
```

## HierarchicalTimeSeriesTransformer 架構

```
                    輸入數據
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    [高頻]     [中頻]       [低頻]
    36點        12點         4點
    (B,36,57) (B,12,57)   (B,4,57)
        │           │           │
        ↓           ↓           ↓
    [Encoder]  [Encoder]   [Encoder]
    (256)      (256)       (256)
        │           │           │
        └───────────┼───────────┘
                    ↓
            [Concatenate]
            (B, 768)
            [256×3]
                    ↓
            [Linear Projection]
            (B, 256)
                    ↓
            [Output]
```

## AttentionWeightedTransformer 架構

```
                    輸入
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    [Expert1]   [Expert2]   [Expert3]
    Transformer Transformer Transformer
        │           │           │
        └───────────┼───────────┘
                    ↓
            [堆疊] (B, 3, 256)
                    │
        ┌───────────┘
        ↓
    [第一個Expert]
    (B, 256)
        │
        ↓
    [Gate機制]
    Linear(256→128)
    ReLU
    Linear(128→3)
    Softmax
    
    權重: w₁, w₂, w₃ (和=1)
        │
        ↓
    [加權平均]
    output = Σ(wᵢ × expertᵢ)
    (B, 256)
```

## 訓練流程圖

```
Epoch 1
    │
    ├─ Batch 1
    │   ├─ Forward Pass
    │   ├─ Loss Calculation
    │   ├─ Backward Pass
    │   └─ Optimizer Step
    │
    ├─ Batch 2, 3, ..., N
    │   └─ [重複]
    │
    ├─ Validation
    │   ├─ Forward Pass
    │   └─ 計算 val_loss
    │
    ├─ 檢查 val_loss
    │   ├─ 若改進 → 保存檢查點
    │   └─ 若未改進 patience++
    │
    └─ 提前停止檢查
        ├─ 若 patience >= 6 → 停止
        └─ 否則繼續 Epoch 2

完成
    │
    └─ 加載最佳檢查點
```

## 梯度流動 (反向傳播)

```
Loss = 1.0*ce_loss + 0.5*mse_loss + 0.2*vol_loss + 0.5*tp_loss
    │
    ├─ d(Loss)/d(cls_logits)
    ├─ d(Loss)/d(reg)
    ├─ d(Loss)/d(vol)
    └─ d(Loss)/d(tp)
    │
    ↓ [通過 MultiTaskHeads]
    │
    ├─ d(Loss)/d(d_cls_input)
    ├─ d(Loss)/d(d_reg_input)
    ├─ d(Loss)/d(d_vol_input)
    └─ d(Loss)/d(d_tp_input)
    │
    ↓ [加總至主要表示]
    │
    d(Loss)/d(backbone_output)
    │
    ↓ [通過 Fusion Layer (若有 symbol)]
    │
    ├─ d(Loss)/d(backbone_hidden)
    └─ d(Loss)/d(symbol_emb)
    │
    ↓ [通過 Backbone]
    │
    d(Loss)/d(pos_emb)
    d(Loss)/d(attention_weights)
    d(Loss)/d(feedforward_params)
    │
    ↓ [梯度裁剪: clip_norm=1.0]
    │
    [梯度更新] (AdamW optimizer)
```

## 損失函數組成

```
Total Loss
    │
    ├─ Classification Loss (CE) × 1.0
    │   └─ CrossEntropyLoss([pred_cls], [true_cls])
    │
    ├─ Regression Loss (MSE) × 0.5
    │   └─ MSELoss([pred_reg], [true_reg])
    │
    ├─ Volatility Loss (MSE) × 0.2
    │   └─ MSELoss([pred_vol], [true_vol])
    │
    └─ TP/SL Loss (MSE) × 0.5
        └─ MSELoss([pred_tp], [true_tp])

權重表示各任務的重要性：
- 1.0: 分類最重要 (降風險)
- 0.5: 迴歸和 TP 次要 (提收益)
- 0.2: 波動最輔助 (增穩定性)
```

## 內存分配

```
模型參數:              800K-2.4M
├─ Backbone:          500K-1M
├─ Symbol Emb:        7K (7×32)
├─ Fusion:            70K (288→256)
└─ Heads:             100K-500K

批量梯度:             50-200MB (取決於 batch_size)
├─ forward 激活值:    30-100MB
└─ 反向傳播中間值:    20-100MB

檢查點:               4-10MB (模型狀態)

總計:                 100-300MB (典型配置)
```

## 推理延遲

```
單個樣本推理時間：
時序投影:     0.1ms
位置編碼:     <0.1ms
Encoder:      10-50ms (取決於層數)
Symbol Emb:   <0.1ms
Fusion:       0.1ms
Heads:        0.5ms
─────────────────────
總計:         10-60ms (CPU) 或 1-5ms (GPU)

Batch 大小 256 推理：
預期吞吐量:   5000-10000 樣本/秒 (GPU)
             100-500 樣本/秒 (CPU)
```

---

**說明**：上述圖表展示了完整的 Transformer 架構、數據流和計算流程。
