import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging

# ========================================================================================
# 【基礎 Transformer Backbone】
# ========================================================================================

class TimeSeriesTransformer(nn.Module):
	"""
	基礎 Transformer Encoder (時序專用)
	
	設計特點：
	- Input: (batch, seq_len, input_dim) → 時序 token 序列
	- Sinusoidal Positional Embedding（時間位置編碼）
	- Transformer Encoder（純注意力機制）
	- Output: (batch, d_model) → 最後一個時間點的 representation
	
	使用場景：
	- 多變量時序分類/回歸
	- 時間序列長期依賴捕捉
	"""
	def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=4, seq_len=36, dropout=0.1):
		super().__init__()
		self.input_dim = input_dim
		self.d_model = d_model
		self.seq_len = seq_len
		
		# Linear projection: input_dim → d_model
		self.in_proj = nn.Linear(input_dim, d_model)
		
		# Learnable positional embedding (簡單但有效)
		# 也可改為 sinusoidal positional encoding
		self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
		nn.init.normal_(self.pos_emb, std=0.02)
		
		# Transformer Encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			batch_first=True,
			dim_feedforward=d_model*4,
			dropout=dropout,
			activation='gelu'
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""
		Args:
			x: (batch, seq_len, input_dim) - 時序特徵
		
		Returns:
			h: (batch, d_model) - 壓縮後的市場狀態表示
		"""
		# x: (batch, seq_len, input_dim)
		x = self.in_proj(x)  # (batch, seq_len, d_model)
		
		# 加上位置編碼
		x = x + self.pos_emb[:, :x.size(1)]
		x = self.dropout(x)
		
		# Encoder (純注意力)
		h = self.encoder(x)  # (batch, seq_len, d_model)
		
		# 取最後一個時間點作為「當前市場狀態」
		return h[:, -1]  # (batch, d_model)
	
# ========================================================================================
# 【Symbol Embedding + Multi-task Heads + Full Model】
# ========================================================================================

class SymbolEmbedding(nn.Module):
	"""
	Simple symbol embedding layer: maps discrete symbol indices to dense vectors.
	"""
	def __init__(self, n_symbols: int, embedding_dim: int = 32):
		super().__init__()
		self.embedding = nn.Embedding(n_symbols, embedding_dim)
		self.embedding_dim = embedding_dim

	def forward(self, symbol_idx: torch.Tensor) -> torch.Tensor:
		"""symbol_idx: (batch,) long tensor -> (batch, embedding_dim)"""
		return self.embedding(symbol_idx)


class MultiTaskHeads(nn.Module):
	"""
	Multi-task heads for classification/regression/volatility/tp outputs.
	Each head is optional (create only when requested). If use_intermediate=True,
	heads are small MLPs; otherwise single linear layers.
	"""
	def __init__(self, d_model: int, n_cls: int = 3, n_tp: int = 3,
				 use_intermediate: bool = True, dropout: float = 0.1,
				 use_cls: bool = True, use_reg: bool = True,
				 use_vol: bool = True, use_tp: bool = True):
		super().__init__()
		mid = d_model // 2 if use_intermediate else d_model

		# Create only requested heads; missing heads will not appear in forward output
		self.use_cls = use_cls
		self.use_reg = use_reg
		self.use_vol = use_vol
		self.use_tp = use_tp

		if use_intermediate:
			if use_cls:
				self.cls_head = nn.Sequential(
					nn.Linear(d_model, mid),
					nn.ReLU(),
					nn.Dropout(dropout),
					nn.Linear(mid, n_cls)
				)
			if use_reg:
				self.reg_head = nn.Sequential(
					nn.Linear(d_model, mid),
					nn.ReLU(),
					nn.Dropout(dropout),
					nn.Linear(mid, 1)
				)
			if use_vol:
				self.vol_head = nn.Sequential(
					nn.Linear(d_model, mid),
					nn.ReLU(),
					nn.Dropout(dropout),
					nn.Linear(mid, 1)
				)
			if use_tp:
				self.tp_head = nn.Sequential(
					nn.Linear(d_model, mid),
					nn.ReLU(),
					nn.Dropout(dropout),
					nn.Linear(mid, n_tp)
				)
		else:
			if use_cls:
				self.cls_head = nn.Linear(d_model, n_cls)
			if use_reg:
				self.reg_head = nn.Linear(d_model, 1)
			if use_vol:
				self.vol_head = nn.Linear(d_model, 1)
			if use_tp:
				self.tp_head = nn.Linear(d_model, n_tp)

	def forward(self, h: torch.Tensor) -> dict:
		"""h: (batch, d_model) -> dict of outputs"""
		out = {}
		if self.use_cls:
			out['cls'] = self.cls_head(h)
		if self.use_reg:
			out['reg'] = self.reg_head(h).squeeze(-1)
		if self.use_vol:
			out['vol'] = self.vol_head(h).squeeze(-1)
		if self.use_tp:
			out['tp'] = self.tp_head(h)
		return out


class TimeSeriesTransformerModel(nn.Module):
	"""
	Full model: backbone (TimeSeriesTransformer) + optional symbol embedding + multi-task heads.
	"""
	def __init__(self,
				 input_dim: int,
				 d_model: int = 256,
				 n_heads: int = 8,
				 n_layers: int = 4,
				 seq_len: int = 36,
				 n_cls: int = 3,
				 n_tp: int = 3,
				 n_symbols: int = 7,
				 symbol_embedding_dim: int = 32,
				 use_symbol_emb: bool = True,
				 use_intermediate_heads: bool = True,
				 use_cls: bool = True,
				 use_reg: bool = True,
				 use_vol: bool = True,
				 use_tp: bool = True):
		super().__init__()
		self.backbone = TimeSeriesTransformer(input_dim, d_model, n_heads, n_layers, seq_len)
		self.d_model = d_model
		self.use_symbol_emb = use_symbol_emb

		if use_symbol_emb:
			self.symbol_emb = SymbolEmbedding(n_symbols, symbol_embedding_dim)
			self.fusion = nn.Sequential(
				nn.Linear(d_model + symbol_embedding_dim, d_model),
				nn.ReLU(),
				nn.Dropout(0.1)
			)
		else:
			self.symbol_emb = None
			self.fusion = None
		# Create only the heads requested
		self.heads = MultiTaskHeads(d_model, n_cls=n_cls, n_tp=n_tp,
					    use_intermediate=use_intermediate_heads,
					    use_cls=use_cls, use_reg=use_reg,
					    use_vol=use_vol, use_tp=use_tp)

	def forward(self, x: torch.Tensor, symbol_idx: torch.Tensor = None) -> dict:
		"""x: (batch, seq_len, input_dim)
		   symbol_idx: (batch,) long tensor if use_symbol_emb
		"""
		h = self.backbone(x)  # (batch, d_model)
		if self.use_symbol_emb and symbol_idx is not None:
			se = self.symbol_emb(symbol_idx)
			h = torch.cat([h, se], dim=-1)
			if self.fusion is not None:
				h = self.fusion(h)

		return self.heads(h)

def train_transformer(
		model,
		train_loader,
		val_loader,
		device,
		task_heads,
		epochs=50,
		patience=6,
		lr=1e-3,
		weight_decay=1e-4,
		loss_weights=None,
		ckpt="transformer.ckpt"):
	"""
	Args:
		model: TimeSeriesTransformerModel
		train_loader, val_loader: DataLoader
		device: torch.device
		task_heads: list, e.g. ["cls", "reg"]
		loss_weights: dict, e.g. {"cls":1.0, "reg":0.5, "vol":0.2, "tp":0.5}
	"""
	opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	# 自動計算 class weight
	class_weight = None
	if "cls" in task_heads:
		# 收集所有 y_cls 標籤
		all_cls_labels = []
		for batch in train_loader:
			y_cls = batch[1].detach().cpu().numpy()
			# 只收集有效標籤（>=0）
			all_cls_labels.extend(y_cls[y_cls >= 0].tolist())
		if all_cls_labels:
			labels_arr = np.array(all_cls_labels)
			n_classes = int(labels_arr.max()) + 1
			counts = np.bincount(labels_arr.astype(int), minlength=n_classes)
			freq = counts / counts.sum()
			# inverse frequency
			inv_freq = 1.0 / (freq + 1e-8)
			# normalize
			norm_inv_freq = inv_freq / inv_freq.sum()
			class_weight = torch.tensor(norm_inv_freq, dtype=torch.float32)
			# print("class_weights:", class_weight.tolist())

	ce = nn.CrossEntropyLoss(reduction="none", weight=class_weight.to(device) if class_weight is not None else None)
	smoothl1 = nn.SmoothL1Loss(reduction="none")
	# mse = nn.MSELoss(reduction="none")
	best_val = float("inf"); best_epoch = -1
	scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
	if loss_weights is None:
		loss_weights = {"cls":1.0, "reg":0.5, "vol":0.2, "tp":0.5}

	for ep in range(1, epochs+1):
		model.train()
		tr_loss = 0.0; ntr = 0
		for batch in train_loader:
			xb = batch[0].to(device)
			y_cls = batch[1].to(device)
			y_reg = batch[2].to(device)
			# dataset now returns y_vol and y_tp_idx before weight
			y_vol = batch[3].to(device)
			y_tp = batch[4].to(device)
			w = batch[5].to(device) if len(batch) > 5 else torch.ones_like(y_cls, dtype=torch.float32)

			# 檢查批次是否有 NaN
			if torch.isnan(xb).any():
				logging.warning(f"NaN in xb (features); skipping batch @ epoch {ep}")
				continue
			if "cls" in task_heads and torch.isnan(y_cls).any():
				logging.warning(f"NaN in y_cls; skipping batch @ epoch {ep}")
				continue
			if "reg" in task_heads and torch.isnan(y_reg).any():
				logging.warning(f"NaN in y_reg; skipping batch @ epoch {ep}")
				continue
			if "vol" in task_heads and torch.isnan(y_vol).any():
				logging.warning(f"NaN in y_vol; skipping batch @ epoch {ep}")
				continue
			if "tp" in task_heads and torch.isnan(y_tp).any():
				logging.warning(f"NaN in y_tp; skipping batch @ epoch {ep}")
				continue
			if torch.isnan(w).any():
				logging.warning(f"NaN in w (weights); skipping batch @ epoch {ep}")
				continue

			opt.zero_grad(set_to_none=True)
			# Use autocast only when CUDA is available (match LSTM training)
			# prepare placeholders for diagnostics
			loss_cls = loss_reg = loss_vol = loss_tp = None
			with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type="cuda"):
				# xb_cpu = xb.detach().cpu()
				# print(f" xb stats: min={float(xb_cpu.min()):.3e} max={float(xb_cpu.max()):.3e} mean={float(xb_cpu.mean()):.3e} std={float(xb_cpu.std()):.3e}")
				# print(f" xb finite: {int(torch.isfinite(xb).sum().item())}/{xb.numel()}")
				# print(f" xb >1e3: {(xb_cpu.abs() > 1e3).sum().item()}  >1e6: {(xb_cpu.abs() > 1e6).sum().item()}")
				# xb = xb.clamp(-1e2, 1e2)
				out = model(xb)
				loss = 0.0
				if "cls" in task_heads:
					mask_cls = y_cls >= 0
					if mask_cls.any():
						loss_cls = (ce(out["cls"][mask_cls], y_cls[mask_cls]) * w[mask_cls]).mean()
						loss = loss + loss_weights["cls"] * loss_cls
				if "reg" in task_heads:
					mask_reg = torch.isfinite(y_reg)
					if mask_reg.any():
						loss_reg = (smoothl1(out["reg"][mask_reg], y_reg[mask_reg]) * w[mask_reg]).mean()
						loss = loss + loss_weights["reg"] * loss_reg
				# vol (regression)
				if "vol" in task_heads:
					mask_vol = torch.isfinite(y_vol)
					if mask_vol.any():
						loss_vol = (smoothl1(out["vol"][mask_vol], y_vol[mask_vol]) * w[mask_vol]).mean()
						loss = loss + loss_weights.get("vol", 0.0) * loss_vol
				# tp (classification)
				if "tp" in task_heads:
					mask_tp = y_tp >= 0
					if mask_tp.any():
						loss_tp = (ce(out["tp"][mask_tp], y_tp[mask_tp]) * w[mask_tp]).mean()
						loss = loss + loss_weights.get("tp", 0.0) * loss_tp
			# guard against NaN loss and print diagnostics
			if torch.isnan(loss):
				logging.error(f"NaN loss encountered; dumping diagnostics @ epoch {ep}")
				try:
					logging.error(f"batch shapes: xb={tuple(xb.shape)}, out[cls]={tuple(out['cls'].shape) if 'cls' in out else None}, out[reg]={tuple(out['reg'].shape) if 'reg' in out else None}")
					# per-head loss summaries
					if loss_cls is not None:
						logging.error(f"loss_cls: mean={loss_cls.detach().cpu().item():.6f}")
					if loss_reg is not None:
						logging.error(f"loss_reg: mean={loss_reg.detach().cpu().item():.6f}")
					if loss_vol is not None:
						logging.error(f"loss_vol: mean={loss_vol.detach().cpu().item():.6f}")
					if loss_tp is not None:
						logging.error(f"loss_tp: mean={loss_tp.detach().cpu().item():.6f}")
					# model outputs finite stats
					for k in out:
						v = out[k]
						finite = torch.isfinite(v)
						logging.error(f"out[{k}]: shape={tuple(v.shape)} finite_count={int(finite.sum().item())}/{v.numel()}")
					# target finite stats
					logging.error(f"y_reg finite: {int(torch.isfinite(y_reg).sum().item())}/{y_reg.numel()}")
					logging.error(f"y_vol finite: {int(torch.isfinite(y_vol).sum().item())}/{y_vol.numel()}")
					logging.error(f"weights finite: {int(torch.isfinite(w).sum().item())}/{w.numel()}")
				except Exception:
					logging.exception("diagnostic print failed")
				continue
			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(opt)
			scaler.update()
			tr_loss += loss.item() * xb.size(0)
			ntr += xb.size(0)
		tr_loss = tr_loss / max(ntr,1) if ntr > 0 else float('nan')

		# Validation
		model.eval()
		with torch.no_grad():
			va_loss = 0.0; nva = 0
			for batch in val_loader:
				xb = batch[0].to(device)
				out = model(xb)
				loss = 0.0
				if "cls" in task_heads:
					y_cls = batch[1].to(device)
					mask_cls = y_cls >= 0
					if mask_cls.any():
						loss += loss_weights["cls"] * ce(out["cls"][mask_cls], y_cls[mask_cls]).mean()
				if "reg" in task_heads:
					y_reg = batch[2].to(device)
					mask_reg = torch.isfinite(y_reg)
					if mask_reg.any():
						loss += loss_weights["reg"] * smoothl1(out["reg"][mask_reg], y_reg[mask_reg]).mean()
				if "vol" in task_heads:
					y_vol = batch[3].to(device)
					mask_vol = torch.isfinite(y_vol)
					if mask_vol.any():
						loss += loss_weights.get("vol", 0.0) * smoothl1(out["vol"][mask_vol], y_vol[mask_vol]).mean()
				if "tp" in task_heads:
					y_tp = batch[4].to(device)
					mask_tp = y_tp >= 0
					if mask_tp.any():
						loss += loss_weights.get("tp", 0.0) * ce(out["tp"][mask_tp], y_tp[mask_tp]).mean()
				va_loss += loss.item() * xb.size(0); nva += xb.size(0)
			va_loss /= max(nva,1)

		logging.info(f"[epoch {ep}] train={tr_loss:.6f} val={va_loss:.6f} best={best_val:.6f}@{best_epoch}")
		if va_loss < best_val - 1e-6:
			best_val = va_loss; best_epoch = ep
			torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
		elif ep - best_epoch >= patience:
			logging.info(f"[early-stop] best={best_val:.6f} @ epoch {best_epoch}")
			break
		
		# if ep == 1:
		# 	break

	ck = torch.load(ckpt, map_location=device)
	model.load_state_dict(ck["model"])
	return model

# ----------------------
# Evaluation (mirror train_lstm.evaluate)
# ----------------------
def evaluate_transformer(model, loader, device, task_heads):
	model.eval()
	out = {}
	with torch.no_grad():
		correct_cls = 0; total_cls = 0
		reg_sum = 0.0; reg_cnt = 0
		vol_sum = 0.0; vol_cnt = 0
		correct_tp = 0; total_tp = 0
		reg_pred_list = []
		reg_true_list = []
		for batch in loader:
			xb = batch[0].to(device)
			yb_cls = batch[1].to(device)
			yb_reg = batch[2].to(device)
			yb_vol = batch[3].to(device)
			yb_tp = batch[4].to(device)
			# forward
			preds = model(xb)
			# classification (main)
			if "cls" in task_heads:
				mask_cls = yb_cls >= 0
				if mask_cls.any():
					pred = preds["cls"][mask_cls].argmax(dim=1)
					correct_cls += (pred == yb_cls[mask_cls]).sum().item()
					total_cls += mask_cls.sum().item()
			# regression
			if "reg" in task_heads:
				mask_reg = torch.isfinite(yb_reg)
				if mask_reg.any():
					# 將預測值與真實值都除以100，確保評估在原始尺度
					reg_pred = preds["reg"][mask_reg] / 100.0
					reg_true = yb_reg[mask_reg] / 100.0
					reg_sum += torch.abs(reg_pred - reg_true).sum().item()
					reg_cnt += mask_reg.sum().item()
					# 收集預測與真實值
					reg_pred_list.append(reg_pred.detach().cpu().numpy())
					reg_true_list.append(reg_true.detach().cpu().numpy())
			# vol (regression)
			if "vol" in task_heads:
				mask_vol = torch.isfinite(yb_vol)
				if mask_vol.any():
					vol_sum += torch.abs(preds["vol"][mask_vol] - yb_vol[mask_vol]).sum().item()
					vol_cnt += mask_vol.sum().item()
			# tp (classification)
			if "tp" in task_heads:
				mask_tp = yb_tp >= 0
				if mask_tp.any():
					pred_tp = preds["tp"][mask_tp].argmax(dim=1)
					correct_tp += (pred_tp == yb_tp[mask_tp]).sum().item()
					total_tp += mask_tp.sum().item()

	if "cls" in task_heads:
		out["acc"] = correct_cls / total_cls if total_cls > 0 else float("nan")
	if "reg" in task_heads:
		out["mae"] = reg_sum / reg_cnt if reg_cnt > 0 else float("nan")
		# 計算 Information Coefficient (IC)
		if reg_pred_list and reg_true_list:
			reg_pred_all = np.concatenate(reg_pred_list)
			reg_true_all = np.concatenate(reg_true_list)
			if len(reg_pred_all) == len(reg_true_all) and len(reg_pred_all) > 1:
				ic = np.corrcoef(reg_pred_all, reg_true_all)[0,1]
				out["ic"] = float(ic)
			else:
				out["ic"] = float("nan")
		else:
			out["ic"] = float("nan")
	if "vol" in task_heads:
		out["vol_mae"] = vol_sum / vol_cnt if vol_cnt > 0 else float("nan")
	if "tp" in task_heads:
		out["tp_acc"] = correct_tp / total_tp if total_tp > 0 else float("nan")
	return out

# ========================================================================================
# 【進階變體：分層 + 注意力加權】
# ========================================================================================


# class HierarchicalTimeSeriesTransformer(nn.Module):
# 	"""
# 	分層時序 Transformer (多尺度)
# 	
# 	設計：
# 	- 多個 Encoder (粗到細: 30min/1h/4h/1d)
# 	- 各層獨立學習不同時間尺度
# 	- 最後融合得到多尺度表示
# 	
# 	優勢：
# 	- 同時捕捉短期波動 + 長期趨勢
# 	- 更強的特徵學習能力
# 	- 適合多時間框架交易策略
# 	"""
# 	def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=4, seq_lens=[36, 12, 4]):
# 		super().__init__()
# 		self.encoders = nn.ModuleList()
# 		self.d_model = d_model
# 		for seq_len in seq_lens:
# 			enc = TimeSeriesTransformer(input_dim, d_model, n_heads, n_layers, seq_len)
# 			self.encoders.append(enc)
# 		# 融合層
# 		self.fusion = nn.Linear(d_model * len(seq_lens), d_model)
# 	
# 	def forward(self, xs):
# 		"""
# 		Args:
# 			xs: list of (batch, seq_len_i, input_dim) - 多尺度時序
# 		
# 		Returns:
# 			h: (batch, d_model) - 融合表示
# 		"""
# 		hs = [enc(x) for enc, x in zip(self.encoders, xs)]
# 		h_cat = torch.cat(hs, dim=-1)
# 		return self.fusion(h_cat)
# 
# 
# class AttentionWeightedTimeSeriesTransformer(nn.Module):
# 	"""
# 	注意力加權時序 Transformer
# 	
# 	設計：
# 	- 多個 Encoder (不同初始化/超參數)
# 	- 用可學習的注意力權重融合
# 	- 自動發現最優的特徵融合策略
# 	
# 	優勢：
# 	- 集成學習：多個視角 → 更穩健
# 	- 自適應加權：數據驅動
# 	- 可視化：注意力權重反映重要性
# 	"""
# 	def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=4, seq_len=36, n_experts=3):
# 		super().__init__()
# 		self.n_experts = n_experts
# 		self.experts = nn.ModuleList([
# 			TimeSeriesTransformer(input_dim, d_model, n_heads, n_layers, seq_len)
# 			for _ in range(n_experts)
# 		])
# 		# 可學習的 gate (注意力)
# 		self.gate = nn.Sequential(
# 			nn.Linear(d_model, d_model // 2),
# 			nn.ReLU(),
# 			nn.Linear(d_model // 2, n_experts),
# 			nn.Softmax(dim=-1)
# 		)
# 	
# 	def forward(self, x):
# 		"""
# 		Args:
# 			x: (batch, seq_len, input_dim)
# 		
# 		Returns:
# 			h: (batch, d_model) - 加權融合的表示
# 		"""
# 		hs = [exp(x) for exp in self.experts]  # list of (batch, d_model)
# 		hs = torch.stack(hs, dim=1)  # (batch, n_experts, d_model)
# 		
# 		# 計算注意力權重
# 		# 使用第一個 expert 的輸出來計算 gate (也可用其他方式)
# 		gate_input = hs[:, 0]  # (batch, d_model)
# 		w = self.gate(gate_input)  # (batch, n_experts)
# 		
# 		# 加權平均
# 		h = (hs * w.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
# 		return h

