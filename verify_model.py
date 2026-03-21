#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_model.py

簡潔說明:
- 載入 checkpoint、重建模型、做 inference（production 或 backtest）
- 支援 `--mode backtest`（historical data + labels）與 `--mode online`（features only）

使用範例:
    python verify_model.py --ckpt <CKPT> --mode online --symbols BTCUSDT --start 2025-09-01 --end 2025-10-26 --interval 30
"""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tool.data_loader import LazySeqDataset
from tool.transformer import TimeSeriesTransformerModel, evaluate_transformer
from train_transformer import load_merged_months, normalize_features_transformer, drop_labels_from_features, read_monthly, month_span

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ----------------------
# Helper: Print model summary
# ----------------------
def print_model_summary(model, input_shape, device):
    """Print model architecture and parameter count."""
    logger.info("\n" + "="*80)
    logger.info("【MODEL ARCHITECTURE】")
    logger.info("="*80)
    
    # Print model structure
    logger.info(f"\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass shape
    logger.info(f"\n【Expected Input Shape】")
    logger.info(f"  x (features): {input_shape}")
    logger.info(f"  symbol_idx (optional): (batch,) long tensor")
    
    logger.info(f"\n【Expected Output】")
    logger.info(f"  Dict with keys (if enabled):")
    logger.info(f"    - cls: (batch, n_classes) logits")
    logger.info(f"    - reg: (batch,) regression values")
    logger.info(f"    - vol: (batch,) volatility values")
    logger.info(f"    - tp: (batch, n_tp_classes) logits")
    
    logger.info("="*80 + "\n")


def test_forward_pass(model, device, input_shape, n_features, n_symbols=None):
    """Test forward pass with dummy data."""
    logger.info("\n【Testing forward pass with dummy data】")
    
    try:
        batch_size, seq_len, _ = input_shape
        
        # Create dummy input
        dummy_x = torch.randn(batch_size, seq_len, n_features, device=device)
        
        # Test with symbol_idx=None to avoid index out of bounds issues
        # (symbol embedding is optional for inference)
        with torch.no_grad():
            outputs = model(dummy_x, symbol_idx=None)
        
        logger.info("✓ Forward pass successful!")
        logger.info(f"  Outputs:")
        for key, val in outputs.items():
            logger.info(f"    - {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        
        return outputs
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ----------------------
# Load model from checkpoint
# ----------------------
def infer_model_arch_from_ckpt(ckpt_path: Path):
    """Infer model architecture by analyzing checkpoint weights."""
    logger.info(f"【Inferring model architecture from checkpoint weights】")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    
    # Infer n_layers from encoder.layers.*.* keys
    layer_indices = set()
    for key in state_dict.keys():
        if "backbone.encoder.layers." in key:
            # Extract layer index from "backbone.encoder.layers.X.*"
            parts = key.split("backbone.encoder.layers.")
            if len(parts) > 1:
                layer_idx = int(parts[1].split(".")[0])
                layer_indices.add(layer_idx)
    
    n_layers = max(layer_indices) + 1 if layer_indices else 4
    logger.info(f"  Detected n_layers: {n_layers} (from layer indices: {sorted(layer_indices)})")
    
    # Infer n_symbols from symbol_emb.embedding.weight shape
    n_symbols = 7
    if "symbol_emb.embedding.weight" in state_dict:
        embedding_shape = state_dict["symbol_emb.embedding.weight"].shape
        n_symbols = embedding_shape[0]
        logger.info(f"  Detected n_symbols: {n_symbols} (from embedding shape: {embedding_shape})")
    
    return n_layers, n_symbols


def load_model_from_ckpt(ckpt_path: Path, device):
    """Load model architecture and weights from checkpoint and metadata."""
    logger.info(f"\n【Loading checkpoint from {ckpt_path}】")
    
    run_dir = ckpt_path.parent
    metadata_path = run_dir / "metadata.json"
    feature_list_path = run_dir / "feature_list.txt"
    class_map_path = run_dir / "class_map.json"
    
    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Metadata loaded from {metadata_path}")
    logger.info(f"  seq_len: {metadata['seq_len']}")
    logger.info(f"  n_features: {metadata['n_features']}")
    logger.info(f"  n_classes: {metadata['n_classes']}")
    logger.info(f"  n_tp_classes: {metadata['n_tp_classes']}")
    logger.info(f"  task_heads: {metadata['task_heads']}")
    logger.info(f"  model_arch: d_model={metadata.get('d_model', 256)}, n_heads={metadata.get('n_heads', 8)}, n_layers={metadata.get('n_layers', 4)}, n_symbols={metadata.get('n_symbols', 7)}")
    
    # Load feature list
    features = []
    if feature_list_path.exists():
        with open(feature_list_path, "r", encoding="utf-8") as f:
            features = [line.strip() for line in f if line.strip()]
        logger.info(f"✓ Feature list loaded: {len(features)} features")
    
    # Load class maps
    class_maps = {}
    if class_map_path.exists():
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_maps = json.load(f)
        # Normalize class maps to index->label form (keys are string indices: "0","1",...)
        def _normalize_map(m: dict):
            # Simplified normalization for expected shapes like:
            # {"-1": 0, "0": 1, "1": 2}  (label->index)
            # or {"0": -1, "1": 0, "2": 1}  (index->label)
            # If values are 0..N-1 -> invert to index->label
            try:
                vals = [int(float(v)) for v in m.values()]
            except Exception:
                return {str(k): v for k, v in m.items()}

            if set(vals) == set(range(len(m))):
                inv = {}
                for k, v in m.items():
                    idx = str(int(float(v)))
                    # parse original key as int if possible else float
                    try:
                        key_num = int(k)
                    except Exception:
                        try:
                            key_num = float(k)
                        except Exception:
                            key_num = k
                    inv[idx] = key_num
                return inv

            # If keys already look like 0..N-1, normalize values types
            try:
                keys = [int(float(k)) for k in m.keys()]
                if set(keys) == set(range(len(m))):
                    out = {str(int(float(k))): (int(float(v)) if str(v).replace('.','',1).lstrip('-').isdigit() else v)
                           for k, v in m.items()}
                    return out
            except Exception:
                pass

            return {str(k): v for k, v in m.items()}

        class_maps = {k: _normalize_map(v) for k, v in class_maps.items()}
        logger.info(f"✓ Class maps loaded (normalized)")
        if "cls" in class_maps:
            logger.info(f"  cls map: {class_maps['cls']}")
        if "tp" in class_maps:
            logger.info(f"  tp map: {class_maps['tp']}")
    
    # Determine which heads to enable based on metadata
    task_heads = metadata.get("task_heads", ["cls", "reg"])
    
    # Get model architecture from metadata (with fallback defaults)
    d_model = metadata.get("d_model", 256)
    n_heads = metadata.get("n_heads", 8)
    n_layers_meta = metadata.get("n_layers", 4)
    n_symbols_meta = metadata.get("n_symbols", 7)
    
    # Try to infer actual architecture from checkpoint to handle mismatches
    try:
        n_layers_actual, n_symbols_actual = infer_model_arch_from_ckpt(ckpt_path)
        if n_layers_actual != n_layers_meta or n_symbols_actual != n_symbols_meta:
            logger.warning(f"  Metadata mismatch detected!")
            logger.warning(f"    Metadata says: n_layers={n_layers_meta}, n_symbols={n_symbols_meta}")
            logger.warning(f"    Checkpoint has: n_layers={n_layers_actual}, n_symbols={n_symbols_actual}")
            logger.info(f"  Using checkpoint values for loading...")
            n_layers_meta = n_layers_actual
            n_symbols_meta = n_symbols_actual
    except Exception as e:
        logger.warning(f"Could not infer architecture from checkpoint: {e}. Using metadata values.")
    
    logger.info(f"\nRebuilding model with checkpoint architecture:")
    logger.info(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers_meta}, n_symbols={n_symbols_meta}")
    
    # Rebuild model with correct architecture from checkpoint
    model = TimeSeriesTransformerModel(
        input_dim=metadata["n_features"],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers_meta,
        seq_len=metadata["seq_len"],
        n_cls=metadata["n_classes"],
        n_tp=metadata["n_tp_classes"],
        n_symbols=n_symbols_meta,
        symbol_embedding_dim=32,
        use_symbol_emb=True,
        use_intermediate_heads=True,
        use_cls="cls" in task_heads,
        use_reg="reg" in task_heads,
        use_vol="vol" in task_heads,
        use_tp="tp" in task_heads,
    ).to(device)
    
    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"✓ Model weights loaded from epoch {ckpt['epoch']}")
    
    model.eval()
    logger.info(f"✓ Model set to eval mode")
    
    return model, metadata, features, class_maps


# ----------------------
# Create evaluation DataLoader for inference
# ----------------------
def create_eval_dataloader(df_eval, features: list, seq_len: int, batch_size: int = 32,
                          label_cls: str = None, label_reg: str = None,
                          label_vol: str = None, label_tp: str = None):
    """Create evaluation DataLoader from normalized data (no train/val/test split).
    
    Args:
        df_eval: DataFrame with features and optional labels (symbol, open_time required)
        features: list of feature column names
        seq_len: sequence length
        batch_size: batch size for DataLoader
        label_cls, label_reg, label_vol, label_tp: column names for labels (optional)
    
    Returns:
        DataLoader configured for evaluation/inference with or without labels
    """
    # Create targets map - include label column names if provided
    targets_map = {
        "cls": label_cls,
        "reg": label_reg,
        "vol": label_vol,
        "tp": label_tp
    }
    
    # Log what targets are available
    has_labels = any(targets_map.values())
    if has_labels:
        available_targets = [k for k, v in targets_map.items() if v is not None]
        logger.info(f"  Loading labels for: {available_targets}")
    else:
        logger.info(f"  No labels provided (pure inference mode)")
    
    ds_eval = LazySeqDataset(df_eval, features, targets_map, seq_len, weight_mode="none")
    eval_loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"✓ Evaluation DataLoader created: {len(ds_eval)} sequences, batch_size={batch_size}")
    return eval_loader


# ----------------------
# Run inference and evaluation
# ----------------------
def run_inference_and_eval(model, eval_loader, device, task_heads, batch_size: int = 32):
    """DEPRECATED: Use inline code in main() instead.
    
    This function is kept for reference but the main() function now separates
    inference and evaluation into distinct stages for better control.
    """
    pass


# ----------------------
# Summarize predictions
# ----------------------
def summarize_predictions(results: dict, seq_len: int, class_maps: dict = None):
    """Print comprehensive prediction statistics and evaluation metrics.
    
    Args:
        results: dict with "predictions" and "metrics"
        seq_len: sequence length (not used but kept for compatibility)
        class_maps: dict with "cls" and "tp" mappings to convert indices to actual class values
    """
    logger.info(f"\n【PREDICTION STATISTICS】")
    
    preds = results["predictions"]
    
    # Helper function to map logit indices to actual class values
    def map_indices_to_classes(indices, class_map_key):
        """Convert argmax indices to actual class values using class_map."""
        if class_maps and class_map_key in class_maps:
            class_map = class_maps[class_map_key]
            # class_map is typically {"0": -1, "1": 0, "2": 1} or similar
            # We need to map array indices to their string keys, then to actual values
            mapped = np.array([class_map.get(str(idx), idx) for idx in indices])
            return mapped
        return indices
    
    # Classification
    if "cls" in preds and preds["cls"] is not None:
        cls_logits = preds["cls"]
        cls_pred_indices = np.argmax(cls_logits, axis=1)
        cls_pred = map_indices_to_classes(cls_pred_indices, "cls")
        
        logger.info(f"\n【Classification (cls)】 n={len(cls_pred)}")
        unique, counts = np.unique(cls_pred, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  class {u}: {c:5d} ({100*c/len(cls_pred):5.1f}%)")
        logger.info(f"  Logits stats (before argmax):")
        for i in range(cls_logits.shape[1]):
            logger.info(f"    logit_{i}: mean={cls_logits[:, i].mean():.6f}, std={cls_logits[:, i].std():.6f}, "
                       f"range=[{cls_logits[:, i].min():.6f}, {cls_logits[:, i].max():.6f}]")
    
    # Regression
    if "reg" in preds and preds["reg"] is not None:
        reg_pred = preds["reg"]
        logger.info(f"\n【Regression (reg)】 n={len(reg_pred)}")
        logger.info(f"  mean={reg_pred.mean():.6f}, std={reg_pred.std():.6f}")
        logger.info(f"  range=[{reg_pred.min():.6f}, {reg_pred.max():.6f}]")
        logger.info(f"  First 5: {reg_pred[:5]}")
    
    # Volatility
    if "vol" in preds and preds["vol"] is not None:
        vol_pred = preds["vol"]
        logger.info(f"\n【Volatility (vol)】 n={len(vol_pred)}")
        logger.info(f"  mean={vol_pred.mean():.6f}, std={vol_pred.std():.6f}")
        logger.info(f"  range=[{vol_pred.min():.6f}, {vol_pred.max():.6f}]")
        logger.info(f"  First 5: {vol_pred[:5]}")
    
    # Take Profit
    if "tp" in preds and preds["tp"] is not None:
        tp_logits = preds["tp"]
        tp_pred_indices = np.argmax(tp_logits, axis=1)
        tp_pred = map_indices_to_classes(tp_pred_indices, "tp")
        
        logger.info(f"\n【Take Profit (tp)】 n={len(tp_pred)}")
        unique, counts = np.unique(tp_pred, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  class {u}: {c:5d} ({100*c/len(tp_pred):5.1f}%)")
        logger.info(f"  Logits stats (before argmax):")
        for i in range(tp_logits.shape[1]):
            logger.info(f"    logit_{i}: mean={tp_logits[:, i].mean():.6f}, std={tp_logits[:, i].std():.6f}, "
                       f"range=[{tp_logits[:, i].min():.6f}, {tp_logits[:, i].max():.6f}]")
    
    # Evaluation Metrics (if available)
    if "metrics" in results and results["metrics"]:
        logger.info(f"\n【EVALUATION METRICS】")
        has_valid = False
        for key, val in results["metrics"].items():
            if isinstance(val, float):
                if not np.isnan(val):
                    logger.info(f"  {key}: {val:.6f}")
                    has_valid = True
                else:
                    logger.info(f"  {key}: nan")
            else:
                logger.info(f"  {key}: {val}")
        
        if not has_valid:
            logger.warning(f"  ⚠ All metrics are nan - labels were not available or invalid")


def save_predictions(results: dict, output_path: Path, seq_len: int, class_maps: dict = None):
    """Save predictions to CSV.
    
    Args:
        results: dict with "predictions" and "metrics"
        output_path: path to save CSV
        seq_len: sequence length (not used but kept for compatibility)
        class_maps: dict with "cls" and "tp" mappings to convert indices to actual class values
    """
    logger.info(f"\n【Saving predictions to {output_path}】")

    preds = results["predictions"]
    n_preds = max(len(v) for v in preds.values() if v is not None)
    
    # Helper function to map logit indices to actual class values
    def map_indices_to_classes(indices, class_map_key):
        """Convert argmax indices to actual class values using class_map."""
        if class_maps and class_map_key in class_maps:
            class_map = class_maps[class_map_key]
            mapped = np.array([class_map.get(str(idx), idx) for idx in indices])
            return mapped
        return indices
    
    df_out = pd.DataFrame()
    # df_out["sequence_idx"] = np.arange(n_preds)
    df_out["open_time"] = results["open_time"]
    
    if "cls" in preds and preds["cls"] is not None:
        cls_indices = np.argmax(preds["cls"], axis=1)
        cls_mapped = map_indices_to_classes(cls_indices, "cls")
        df_out["cls_pred"] = cls_mapped
        for i in range(preds["cls"].shape[1]):
            df_out[f"cls_logit_{i}"] = preds["cls"][:, i]
    
    if "reg" in preds and preds["reg"] is not None:
        df_out["reg_pred"] = preds["reg"]
    
    if "vol" in preds and preds["vol"] is not None:
        df_out["vol_pred"] = preds["vol"]
    
    if "tp" in preds and preds["tp"] is not None:
        tp_indices = np.argmax(preds["tp"], axis=1)
        tp_mapped = map_indices_to_classes(tp_indices, "tp")
        df_out["tp_pred"] = tp_mapped
        for i in range(preds["tp"].shape[1]):
            df_out[f"tp_logit_{i}"] = preds["tp"][:, i]
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(df_out)} predictions to {output_path}")

def load_raw_data(data_root: Path, exchange: str, symbol: str, times=None) -> pd.DataFrame:
    """讀取 raw data 的 open、high、low、close、volumn

    Args:
        data_root: raw data 根目錄
        exchange: 交易所
        symbol: 交易對
        start: 開始日期 (YYYY-MM-DD)
        end: 結束日期 (YYYY-MM-DD)

    Returns:
        DataFrame with columns: symbol, open_time, open, high, low, close, volume
    """
    logger.info(f"load_raw_data: symbol={symbol} times_provided={times is not None}")

    # If times (open_time list) provided, use them to read specific daily files
    times_list = None
    if times is not None:
        # accept pandas Series, list of datetimes or strings
        if isinstance(times, (pd.Series, list, tuple)):
            times_list = pd.to_datetime(list(times), utc=True)
        else:
            times_list = pd.to_datetime(times, utc=True)
    else:
        logger.warning("No times and no start/end provided — nothing to load")
        return pd.DataFrame()

    # build set of unique dates to load (year, month, day)
    dates_needed = set((int(t.year), int(t.month), int(t.day)) for t in times_list)

    rows = []
    data_root = Path(data_root)
    for y, m, d in sorted(dates_needed):
        day_path = data_root / f"symbol={symbol}" / f"year={y}" / f"month={m:02d}" / f"date={y}-{m:02d}-{d:02d}.parquet"
        if not day_path.exists():
            logger.debug(f"Daily file not found: {day_path}")
            continue
        try:
            day_df = pd.read_parquet(day_path)
        except Exception as e:
            logger.warning(f"Failed to read {day_path}: {e}")
            continue

        if day_df is None or day_df.empty:
            continue

        if "open_time" in day_df.columns:
            day_df["open_time"] = pd.to_datetime(day_df["open_time"], utc=True)
        else:
            logger.debug(f"no open_time in {day_path}")
            continue

        # filter rows where open_time is in requested times_list for this day
        mask = day_df["open_time"].isin(times_list)
        matched = day_df.loc[mask]
        if not matched.empty:
            rows.append(matched)

    if not rows:
        logger.warning("No matching raw rows found for requested times")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # ensure symbol column
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    base_cols = [c for c in ["symbol", "open_time", "open", "high", "low", "close"] if c in df.columns]
    vol_col = None
    for cand in ("volumn", "volume", "qty", "quote_volume"):
        if cand in df.columns:
            vol_col = cand
            break

    out_cols = base_cols + ([vol_col] if vol_col else [])
    df_out = df.loc[:, out_cols].copy()
    if vol_col and vol_col != "volumn":
        df_out = df_out.rename(columns={vol_col: "volumn"})
    if "volumn" not in df_out.columns:
        df_out["volumn"] = 0

    df_out = df_out.sort_values("open_time").reset_index(drop=True)

    # save per-symbol CSV under inference_results/
    out_dir = Path("inference_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_raw.csv"
    try:
        df_out.to_csv(out_path, index=False)
        logger.info(f"Saved raw data for {symbol} to {out_path} ({len(df_out)} rows)")
    except Exception as e:
        logger.warning(f"Failed to save raw CSV to {out_path}: {e}")

    return df_out

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Model verification and inference")
    
    # Model checkpoint
    ap.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint (e.g., logs/train_*/lstm.ckpt)")
    
    # Mode selection: backtest (with labels) or online (without labels)
    ap.add_argument("--mode", type=str, choices=["backtest", "online"], default="backtest",
                   help="backtest: load historical data with labels for evaluation; online: load new data without labels for production inference")
    
    # Data parameters
    ap.add_argument("--features_root", type=str, default="features")
    ap.add_argument("--labels_root", type=str, default="labels")
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--symbols", type=str, default="BTCUSDT",
                   help="Symbol(s) to infer on (comma-separated)")
    ap.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--interval", type=int, default=10, help="Resampling interval (minutes)")
    
    # Inference parameters
    ap.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    ap.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")
    
    # Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
    
    model, metadata, features, class_maps = load_model_from_ckpt(ckpt_path, device)
    
    # Print model summary
    input_shape = (32, metadata["seq_len"], metadata["n_features"])
    print_model_summary(model, input_shape, device)
    
    # Test forward pass (without symbol embedding to avoid index out of bounds)
    test_forward_pass(model, device, input_shape, metadata["n_features"])
    
    # Load real data using load_data (proper handling of symbols and normalization)
    logger.info(f"\n【Loading data in {args.mode.upper()} mode】")
    
    features_root = Path(args.features_root)
    labels_root = Path(args.labels_root)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    
    # ========================================================================
    # BACKTEST MODE: Load historical data with labels for evaluation
    # ========================================================================
    if args.mode == "backtest":
        logger.info(f"【BACKTEST MODE】 - Loading data WITH labels for evaluation")
        
        frames = []
        for sym in symbols:
            logger.info(f"Loading {sym} (features + labels)...")
            df_all = load_merged_months(
                features_root=features_root,
                labels_root=labels_root,
                exchange=args.exchange,
                symbol=sym,
                start=args.start,
                end=args.end,
                interval=args.interval
            )
            if not df_all.empty:
                frames.append(df_all)
        
        if not frames:
            logger.error("No data found for specified date range and symbols")
            return
        
        df_data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
        logger.info(f"✓ Loaded {len(df_data)} rows (with labels)")
        
        # Extract label columns BEFORE normalization
        logger.info(f"\n【Detecting label columns】")
        label_cls = f"y_cls_sign_{args.interval}m" if f"y_cls_sign_{args.interval}m" in df_data.columns else None
        label_reg = f"y_reg_ret_{args.interval}m" if f"y_reg_ret_{args.interval}m" in df_data.columns else None
        label_vol = f"y_vol_{args.interval}m" if f"y_vol_{args.interval}m" in df_data.columns else None
        label_tp = f"y_tp_sl_{args.interval}m" if f"y_tp_sl_{args.interval}m" in df_data.columns else None
        
        labels_found = []
        if label_cls:
            labels_found.append(f"cls: {label_cls}")
        if label_reg:
            labels_found.append(f"reg: {label_reg}")
        if label_vol:
            labels_found.append(f"vol: {label_vol}")
        if label_tp:
            labels_found.append(f"tp: {label_tp}")
        
        if labels_found:
            logger.info(f"✓ Labels found: {', '.join(labels_found)}")
        else:
            logger.info(f"  No labels found in data")
    
    # ========================================================================
    # ONLINE MODE: Load new data without labels (production inference)
    # ========================================================================
    else:  # args.mode == "online"
        logger.info(f"【ONLINE MODE】 - Loading data WITHOUT labels (pure inference)")
        
        frames = []
        # Load features month-by-month (avoid passing None to load_merged_months)
        for sym in symbols:
            logger.info(f"Loading {sym} (features only)...")
            fdfs = []
            # Use train_transformer.month_span to iterate months
            from train_transformer import month_span
            for y, m in month_span(args.start, args.end):
                # use read_monthly to load feature files only
                fdf = read_monthly(features_root, args.exchange, sym, y, m, "features", args.interval)
                if not fdf.empty:
                    fdfs.append(fdf)
            if fdfs:
                df_all = pd.concat(fdfs, ignore_index=True)
                # filter by date range and sort to match load_merged_months behavior
                df_all["open_time"] = pd.to_datetime(df_all["open_time"], utc=True)
                s = pd.to_datetime(args.start, utc=True); e = pd.to_datetime(args.end, utc=True)
                df_all = df_all[(df_all["open_time"] >= s) & (df_all["open_time"] < e)]
                df_all = df_all.sort_values(["symbol", "open_time"]).reset_index(drop=True)
                if not df_all.empty:
                    frames.append(df_all)
        
        if not frames:
            logger.error("No data found for specified date range and symbols")
            return
        
        df_data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
        logger.info(f"✓ Loaded {len(df_data)} rows (features only, no labels)")
        
        # No labels in online mode
        label_cls = label_reg = label_vol = label_tp = None
        labels_found = []
    
    # Process each symbol independently to avoid cross-symbol normalization effects
    logger.info("\n【Per-symbol inference (avoids cross-symbol normalization)}")
    seq_len = metadata["seq_len"]
    out_dir = Path("inference_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_heads = metadata.get("task_heads", ["cls", "reg"])


    for sym in symbols:
        logger.info(f"\n--- Processing symbol: {sym} ---")
        df_sym = df_data.loc[df_data["symbol"] == sym].sort_values("open_time").reset_index(drop=True)
        if df_sym.empty:
            logger.warning(f"No data for symbol {sym}, skipping")
            continue

        # Detect label columns for this symbol (already detected globally)
        # Drop labels, normalize per-symbol, then reattach labels if in backtest
        df_features_only_sym = drop_labels_from_features(df_sym)
        df_normalized_sym = normalize_features_transformer(df_features_only_sym, features, window=500)

        df_reduced_sym = pd.merge(df_normalized_sym, df_sym[["symbol", "open_time"]], on=["symbol", "open_time"], how="inner")

        if args.mode == "backtest":
            if label_cls and label_cls in df_sym.columns:
                df_reduced_sym[label_cls] = df_sym[label_cls].iloc[:len(df_reduced_sym)]
            if label_reg and label_reg in df_sym.columns:
                df_reduced_sym[label_reg] = df_sym[label_reg].iloc[:len(df_reduced_sym)]
            if label_vol and label_vol in df_sym.columns:
                df_reduced_sym[label_vol] = df_sym[label_vol].iloc[:len(df_reduced_sym)]
            if label_tp and label_tp in df_sym.columns:
                df_reduced_sym[label_tp] = df_sym[label_tp].iloc[:len(df_reduced_sym)]

        logger.info(f"  symbol={sym} normalized shape: {df_reduced_sym.shape}")

        # Create inference DataLoader for this symbol
        inference_loader = create_eval_dataloader(
            df_eval=df_reduced_sym,
            features=features,
            seq_len=seq_len,
            batch_size=args.batch_size,
            label_cls=None,
            label_reg=None,
            label_vol=None,
            label_tp=None
        )

        # Run inference for this symbol
        all_preds = {key: [] for key in ["cls", "reg", "vol", "tp"]}
        model.eval()
        with torch.no_grad():
            for batch in inference_loader:
                xb = batch[0].to(device)
                outputs = model(xb, symbol_idx=None)
                for key in outputs:
                    all_preds[key].append(outputs[key].cpu().numpy())

        preds_sym = {}
        for key in all_preds:
            if all_preds[key]:
                arr = np.concatenate(all_preds[key], axis=0)
                # 推論時 regression 預測值除以 100
                if key == "reg" and arr is not None:
                    arr = arr / 100.0
                preds_sym[key] = arr
            else:
                preds_sym[key] = None

        logger.info(f"  ✓ Inference complete for {sym}: {len(preds_sym.get('cls', []))} predictions")

        # === Print per-symbol stats (mean, std, range) for classification logits and regression ===
        # Classification logits
        if preds_sym.get("cls") is not None:
            cls_logits = preds_sym["cls"]
            logger.info(f"\n【Classification (cls)】 n={len(cls_logits)}")
            cls_pred_indices = np.argmax(cls_logits, axis=1)
            # Map to class labels if class_maps available
            def map_indices_to_classes(indices, class_map_key):
                if class_maps and class_map_key in class_maps:
                    class_map = class_maps[class_map_key]
                    mapped = np.array([class_map.get(str(idx), idx) for idx in indices])
                    return mapped
                return indices
            cls_pred = map_indices_to_classes(cls_pred_indices, "cls")
            unique, counts = np.unique(cls_pred, return_counts=True)
            for u, c in zip(unique, counts):
                logger.info(f"  class {u}: {c:5d} ({100*c/len(cls_pred):5.1f}%)")
            logger.info(f"  Logits stats (before argmax):")
            for i in range(cls_logits.shape[1]):
                logger.info(f"    logit_{i}: mean={cls_logits[:, i].mean():.6f}, std={cls_logits[:, i].std():.6f}, range=[{cls_logits[:, i].min():.6f}, {cls_logits[:, i].max():.6f}]")

        # Regression
        if preds_sym.get("reg") is not None:
            reg_pred = preds_sym["reg"]
            logger.info(f"\n【Regression (reg)】 n={len(reg_pred)}")
            logger.info(f"  mean={reg_pred.mean():.6f}, std={reg_pred.std():.6f}")
            logger.info(f"  range=[{reg_pred.min():.6f}, {reg_pred.max():.6f}]")
            logger.info(f"  First 5: {reg_pred[:5]}")

        # Evaluation (backtest) per symbol
        metrics_sym = {}
        if args.mode == "backtest" and labels_found:
            eval_loader_sym = create_eval_dataloader(
                df_eval=df_reduced_sym,
                features=features,
                seq_len=seq_len,
                batch_size=args.batch_size,
                label_cls=label_cls,
                label_reg=label_reg,
                label_vol=label_vol,
                label_tp=label_tp
            )
            metrics_sym = evaluate_transformer(model, eval_loader_sym, device, task_heads)
            logger.info(f"  ✓ Evaluation metrics for {sym}: {metrics_sym}")

        # Build open_time list aligned to sequences for this symbol
        open_times_sym = df_reduced_sym["open_time"].iloc[seq_len - 1:].tolist()

        # Save per-symbol predictions
        sym_results = {"metrics": metrics_sym, "predictions": preds_sym, "open_time": open_times_sym}
        out_path = out_dir / f"{sym}_predictions.csv"
        save_predictions(sym_results, out_path, seq_len, class_maps)

    logger.info(f"\n✓ All symbols processed")
    
    logger.info("\n【Verification complete】\n")

    frames = []
    for sym in symbols:
        logger.info(f"Loading {sym} (features + labels)...")
        # pass the exact open_time values from df_data for this symbol so we extract matching raw rows
        sym_times = df_data.loc[df_data["symbol"] == sym, "open_time"].tolist()[metadata["seq_len"] - 1:]  # skip first seq_len rows which are dropped in dataset
        df_all = load_raw_data(
            data_root=Path("data"),
            exchange=args.exchange,
            symbol=sym,
            times=sym_times,
        )
        if not df_all.empty:
            frames.append(df_all)
    
    if not frames:
        logger.error("No data found for specified date range and symbols")
        return
    
    df_data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)


if __name__ == "__main__":
    main()

# Usage examples (short):
# Backtest (with labels):
#   python verify_model.py --ckpt <CKPT> --mode backtest --symbols BTCUSDT --start 2025-01-01 --end 2025-02-01 --interval 30
# Online (production, features only):
#   python verify_model.py --ckpt <CKPT> --mode online --symbols BTCUSDT --start 2025-09-01 --end 2025-09-02 --interval 30 --batch_size 1

