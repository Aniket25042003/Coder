#!/usr/bin/env python3
"""
Phase A pretrain for the coding/security SLM (MLA + DeepSeekMoE).

Locked defaults follow TRAINING_DECISIONS.md:
  - Qwen2.5-Coder tokenizer (~152K), tied embeddings
  - DeepSeek-V3-style MLA + decoupled RoPE
  - MoE Lite-like ~2.4–3.0B active / ~10–16B total
  - BF16 + FSDP2 + FlashAttention/SDPA + activation checkpointing
  - AdamW, cosine LR, MTP D=1, aux load-balance
  - Phase A: 8K context, ~40B token budget

Smoke test (single GPU):
  python train.py --token_budget 1000000 --max_steps 20 --micro_batch_size 1 \\
      --grad_accum_steps 1 --output_dir ./ckpts/smoke --no_distributed

Multi-GPU (GCP A100/H100):
  torchrun --nproc_per_node=8 train.py \\
      --dataset Aniket200325/coder-pretrain-60gb \\
      --output_dir /data/ckpts/coder-3b-moe \\
      --micro_batch_size 1 --grad_accum_steps 16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    _HAS_FLASH = True
except Exception:  # noqa: BLE001
    _flash_attn_func = None
    _HAS_FLASH = False

try:
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install transformers: pip install -r requirements-train.txt") from e

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install datasets: pip install -r requirements-train.txt") from e

# FSDP2 (API location varies slightly across 2.5–2.6)
_HAS_FSDP2 = False
init_device_mesh = None  # type: ignore
MixedPrecisionPolicy = None  # type: ignore
fully_shard = None  # type: ignore
get_model_state_dict = None  # type: ignore
get_optimizer_state_dict = None  # type: ignore
set_model_state_dict = None  # type: ignore
set_optimizer_state_dict = None  # type: ignore
StateDictOptions = None  # type: ignore
dcp = None  # type: ignore

try:
    from torch.distributed.device_mesh import init_device_mesh
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard  # type: ignore
        from torch.distributed.fsdp import MixedPrecisionPolicy  # type: ignore
    _HAS_FSDP2 = True
except Exception:  # noqa: BLE001
    _HAS_FSDP2 = False

try:
    import wandb as _wandb
except Exception:  # noqa: BLE001
    _wandb = None


LOG = logging.getLogger("coder.train")


# =============================================================================
# 1. Config
# =============================================================================


@dataclass
class ModelConfig:
    vocab_size: int = 151936  # padded from Qwen ~151646
    d_model: int = 2048
    n_layers: int = 28
    n_heads: int = 16
    n_dense_layers: int = 1
    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: int = 0
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    # FFN / MoE
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1408
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    moe_aux_loss_alpha: float = 0.01
    moe_capacity_factor: float = 1.25
    score_func: str = "softmax"  # or sigmoid
    # RoPE
    rope_theta: float = 500_000.0
    rope_factor: float = 1.0
    max_seq_len: int = 8192
    original_seq_len: int = 8192
    # MTP
    mtp_depth: int = 1
    # Misc
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class TrainConfig:
    # Data
    dataset: str = "Aniket200325/coder-pretrain-60gb"
    data_dir: Optional[str] = None
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_rate: float = 0.4
    code_sources: Tuple[str, ...] = ("code", "starcoder", "the-stack", "stack")
    # Optim
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Batch / tokens
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    token_budget: int = 40_000_000_000
    max_steps: Optional[int] = None
    # MTP schedule
    mtp_lambda: float = 0.3
    mtp_lambda_late: float = 0.1
    mtp_lambda_switch_frac: float = 0.75
    # Eval / ckpt
    val_every_tokens: int = 1_000_000_000
    early_stop_tokens: int = 3_000_000_000
    ckpt_every_steps: int = 1000
    ckpt_minutes: float = 30.0
    # System
    output_dir: str = "./ckpts/coder-3b-moe"
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_compile: bool = False
    use_wandb: bool = False
    wandb_project: str = "coder-pretrain"
    log_every_steps: int = 10
    smoke_prompt: str = "def fibonacci(n):\n"
    num_workers: int = 2
    prefetch_factor: int = 2
    resume: Optional[str] = None
    no_distributed: bool = False
    pack_cache_dir: Optional[str] = None
    # Model overrides via nested
    model: ModelConfig = field(default_factory=ModelConfig)


# =============================================================================
# 2. Distributed helpers
# =============================================================================


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def init_distributed(cfg: TrainConfig) -> torch.device:
    if cfg.no_distributed or "LOCAL_RANK" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return torch.device("cuda:0")
        return torch.device("cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device(f"cuda:{local_rank}")


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# =============================================================================
# 3. Model: norms, RoPE, MLA, MoE, MTP
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep norm math in FP32 for stability
        orig_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight.float() * x32).to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], cos/sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """RoPE with optional YaRN-style factor for Phase B CPT."""

    def __init__(self, dim: int, max_seq_len: int, theta: float, factor: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.factor = factor
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if factor != 1.0:
            # Simple NTK-aware stretch: scale frequencies by factor
            inv_freq = inv_freq / factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device=device),
            self.sin_cached[:seq_len].to(device=device),
        )


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    q,k,v: [B, S, H, D]
    Returns: [B, S, H, Dv]
    """
    if _HAS_FLASH and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        # flash_attn expects same head dim for q/k/v; pad v if needed
        dv = v.size(-1)
        dk = k.size(-1)
        if dv != dk:
            v = F.pad(v, (0, dk - dv))
        out = _flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
        if dv != dk:
            out = out[..., :dv]
        return out

    # SDPA path: [B, H, S, D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    dv = v_t.size(-1)
    dk = k_t.size(-1)
    if dv != dk:
        v_t = F.pad(v_t, (0, dk - dv))
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0, is_causal=causal)
    if dv != dk:
        out = out[..., :dv]
    return out.transpose(1, 2)


class MLA(nn.Module):
    """Multi-head Latent Attention with decoupled RoPE (q_lora_rank=0)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.qk_dim = cfg.qk_head_dim

        if cfg.q_lora_rank and cfg.q_lora_rank > 0:
            self.q_a = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
            self.q_b = nn.Linear(cfg.q_lora_rank, cfg.n_heads * self.qk_dim, bias=False)
        else:
            self.q_a = None
            self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.qk_dim, bias=False)

        self.kv_a = nn.Linear(cfg.d_model, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.n_heads

        if self.q_a is not None:
            q = self.q_b(self.q_a_norm(self.q_a(x)))
        else:
            q = self.q_proj(x)
        q = q.view(B, S, H, self.qk_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)
        q_rope = apply_rotary_emb(q_rope, cos[..., : self.qk_rope], sin[..., : self.qk_rope])

        kv = self.kv_a(x)
        compressed, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope], dim=-1)
        compressed = self.kv_a_norm(compressed)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, H, -1)
        k_rope = apply_rotary_emb(k_rope, cos[..., : self.qk_rope], sin[..., : self.qk_rope])

        kv_b = self.kv_b(compressed).view(B, S, H, self.qk_nope + self.v_dim)
        k_nope, v = torch.split(kv_b, [self.qk_nope, self.v_dim], dim=-1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        out = _attention(q, k, v, causal=True)
        out = out.contiguous().view(B, S, H * self.v_dim)
        return self.o_proj(out)


class SwiGLUDenseFFN(nn.Module):
    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, d_model, bias=False)
        self.w3 = nn.Linear(d_model, intermediate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEExpert(nn.Module):
    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, d_model, bias=False)
        self.w3 = nn.Linear(d_model, intermediate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """Shared + fine-grained routed experts with aux load-balancing loss."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.n_activated_experts
        self.alpha = cfg.moe_aux_loss_alpha
        self.capacity_factor = cfg.moe_capacity_factor
        self.score_func = cfg.score_func

        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoEExpert(cfg.d_model, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [MoEExpert(cfg.d_model, cfg.moe_intermediate_size) for _ in range(cfg.n_shared_experts)]
        )
        # Init router near zero for stability
        nn.init.zeros_(self.router.weight)
        self._aux_loss: Optional[torch.Tensor] = None
        self._drop_frac: float = 0.0

    @property
    def aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return torch.tensor(0.0)
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        t = B * S
        flat = x.reshape(t, D)

        # Router in FP32
        logits = self.router(flat.float())
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        if self.score_func == "sigmoid":
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Aux LB: N * sum(f_i * P_i)
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, self.n_routed).float().sum(dim=1)  # [t, E] counts
            tokens_per_expert = one_hot.sum(dim=0) / max(t * self.top_k, 1)
        me = scores.mean(dim=0)
        ce = tokens_per_expert
        self._aux_loss = self.alpha * self.n_routed * (ce * me).sum().to(x.dtype)

        capacity = int(math.ceil(self.capacity_factor * t * self.top_k / self.n_routed))
        capacity = max(capacity, 1)

        out = torch.zeros_like(flat)
        dropped = 0
        total_slots = 0

        for e in range(self.n_routed):
            # Tokens that selected expert e (any of top-k slots)
            mask = (topk_idx == e).any(dim=-1)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            # Weight for this expert: sum of matching topk scores
            weight = torch.zeros(idx.numel(), device=x.device, dtype=topk_scores.dtype)
            for k in range(self.top_k):
                slot = topk_idx[idx, k] == e
                weight = weight + topk_scores[idx, k] * slot.float()

            if idx.numel() > capacity:
                # Keep highest-weight tokens
                keep_scores, keep_pos = torch.topk(weight, capacity)
                dropped += int(idx.numel() - capacity)
                idx = idx[keep_pos]
                weight = keep_scores
            total_slots += int(idx.numel())

            expert_in = flat[idx]
            expert_out = self.experts[e](expert_in)
            out.index_add_(0, idx, expert_out * weight.unsqueeze(-1).to(expert_out.dtype))

        self._drop_frac = float(dropped) / max(t * self.top_k, 1)

        shared_out = sum(se(flat) for se in self.shared)  # type: ignore[misc]
        out = out + shared_out
        return out.view(B, S, D)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = MLA(cfg)
        if layer_id < cfg.n_dense_layers:
            self.ffn: nn.Module = SwiGLUDenseFFN(cfg.d_model, cfg.intermediate_size)
            self.is_moe = False
        else:
            self.ffn = DeepSeekMoE(cfg)
            self.is_moe = True
        # Depth-scaled residual outs applied via init on o_proj / w2

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def collect_aux_loss(self) -> torch.Tensor:
        if self.is_moe and isinstance(self.ffn, DeepSeekMoE):
            return self.ffn.aux_loss
        return torch.tensor(0.0, device=self.attn.o_proj.weight.device)


class MTPHead(nn.Module):
    """Multi-token prediction depth D=1: predict token t+2 from hidden at t."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class CoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.rope = RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
            factor=cfg.rope_factor,
        )
        self.mtp = MTPHead(cfg) if cfg.mtp_depth >= 1 else None
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        self._scale_residuals()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residuals(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for layer in self.layers:
            layer.attn.o_proj.weight.data.mul_(scale)
            if isinstance(layer.ffn, SwiGLUDenseFFN):
                layer.ffn.w2.weight.data.mul_(scale)
            elif isinstance(layer.ffn, DeepSeekMoE):
                for e in list(layer.ffn.experts) + list(layer.ffn.shared):
                    e.w2.weight.data.mul_(scale)
                nn.init.zeros_(layer.ffn.router.weight)

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        # Active ≈ embed + (dense layers full) + MoE layers (shared*all + routed*top_k)/experts + head tied
        cfg = self.cfg
        # Rough FLOP-active estimate
        per_moe_expert = (
            3 * cfg.d_model * cfg.moe_intermediate_size
        )  # w1,w2,w3
        active_moe = (
            cfg.n_shared_experts * per_moe_expert
            + cfg.n_activated_experts * per_moe_expert
        )
        dense_ffn = 3 * cfg.d_model * cfg.intermediate_size
        # Attention params roughly same always-active
        n_moe_layers = cfg.n_layers - cfg.n_dense_layers
        # Use total for reporting; active is estimate excluding unused experts
        unused_experts = n_moe_layers * (cfg.n_routed_experts - cfg.n_activated_experts) * per_moe_expert
        active = total - unused_experts
        return {
            "total_b": total / 1e9,
            "active_est_b": active / 1e9,
            "total": float(total),
            "active_est": float(active),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mtp_lambda: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self.rope(S, h.device)

        aux = h.new_zeros(())
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = activation_checkpoint(
                    layer,
                    h,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            else:
                h = layer(h, cos, sin)
            aux = aux + layer.collect_aux_loss().to(h.device)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = h.new_zeros(())
        ntp_loss = h.new_zeros(())
        mtp_loss = h.new_zeros(())

        if labels is not None:
            # Next-token: predict labels[..., 1:] from logits[..., :-1]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ntp_loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ntp_loss

            if self.mtp is not None and mtp_lambda > 0 and S > 2:
                # Predict token t+2 from hidden at t
                mtp_h = self.mtp(h[:, :-2])
                mtp_logits = self.lm_head(mtp_h)
                mtp_labels = labels[:, 2:].contiguous()
                mtp_loss = F.cross_entropy(
                    mtp_logits.float().view(-1, mtp_logits.size(-1)),
                    mtp_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss + mtp_lambda * mtp_loss

            n_moe = max(self.cfg.n_layers - self.cfg.n_dense_layers, 1)
            loss = loss + aux / n_moe

        return {
            "logits": logits,
            "loss": loss,
            "ntp_loss": ntp_loss.detach(),
            "mtp_loss": mtp_loss.detach(),
            "aux_loss": (aux / max(self.cfg.n_layers - self.cfg.n_dense_layers, 1)).detach(),
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new: int = 64) -> torch.Tensor:
        self.eval()
        ids = input_ids
        for _ in range(max_new):
            out = self.forward(ids)
            next_id = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if ids.size(1) >= self.cfg.max_seq_len:
                break
        return ids


# =============================================================================
# 4. Data: FIM + packing + streaming
# =============================================================================

CODE_SOURCE_HINTS = ("code", "starcoder", "stack", "the-stack", "github")


def load_qwen_tokenizer(name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    required = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    for t in required:
        tid = tok.convert_tokens_to_ids(t)
        if tid is None or tid < 0 or (
            getattr(tok, "unk_token_id", None) is not None and tid == tok.unk_token_id
        ):
            raise RuntimeError(f"Tokenizer {name} missing required FIM token {t!r}")
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer missing eos_token_id")
    return tok


def _is_code_source(source: Optional[str], hints: Tuple[str, ...]) -> bool:
    if not source:
        return False
    s = source.lower()
    return any(h in s for h in hints)


def apply_fim(
    token_ids: List[int],
    tok: Any,
    rng: random.Random,
) -> List[int]:
    """Prefix-Suffix-Middle FIM on token sequence."""
    if len(token_ids) < 8:
        return token_ids
    # Split into prefix / middle / suffix
    n = len(token_ids)
    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)
    prefix, middle, suffix = token_ids[:i], token_ids[i:j], token_ids[j:]
    fim_p = tok.convert_tokens_to_ids("<|fim_prefix|>")
    fim_m = tok.convert_tokens_to_ids("<|fim_middle|>")
    fim_s = tok.convert_tokens_to_ids("<|fim_suffix|>")
    # SPM order common for Qwen: prefix, suffix, middle
    return [fim_p] + prefix + [fim_s] + suffix + [fim_m] + middle


def chunk_long_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    # Prefer newline splits
    buf: List[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and buf:
            parts.append("".join(buf))
            buf, size = [], 0
        if len(line) > max_chars:
            if buf:
                parts.append("".join(buf))
                buf, size = [], 0
            for k in range(0, len(line), max_chars):
                parts.append(line[k : k + max_chars])
            continue
        buf.append(line)
        size += len(line)
    if buf:
        parts.append("".join(buf))
    return parts


@dataclass
class DataCursor:
    epoch: int = 0
    shard_idx: int = 0
    row_offset: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataCursor":
        return cls(
            epoch=int(d.get("epoch", 0)),
            shard_idx=int(d.get("shard_idx", 0)),
            row_offset=int(d.get("row_offset", 0)),
        )


class PackedFIMIterable(IterableDataset):
    """
    Streams docs from Hub or local parquet, tokenizes, packs to seq_len,
    optionally applies FIM on code sources. Each rank strides examples.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        tok: Any,
        split: str = "train",
        cursor: Optional[DataCursor] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.split = split
        self.cursor = cursor or DataCursor()
        self.seq_len = cfg.model.max_seq_len
        self.eos_id = tok.eos_token_id
        self.rng = random.Random(cfg.seed + get_rank())

    def _load_stream(self, epoch: int) -> Any:
        kwargs: Dict[str, Any] = {"split": self.split, "streaming": True}
        # Retry Hub blips
        last_err: Optional[Exception] = None
        for attempt in range(5):
            try:
                if self.cfg.data_dir:
                    pattern = str(Path(self.cfg.data_dir) / f"**/{self.split}*.parquet")
                    # Also try final/train style
                    paths = list(Path(self.cfg.data_dir).rglob("*.parquet"))
                    if self.split == "val":
                        paths = [p for p in paths if "val" in p.as_posix().lower()]
                    else:
                        paths = [p for p in paths if "val" not in p.as_posix().lower()]
                    if not paths:
                        raise FileNotFoundError(f"No parquet under {self.cfg.data_dir} for split={self.split}")
                    ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train", streaming=True)
                else:
                    # Prefer named split; fall back to train[:]/ / data files
                    try:
                        ds = load_dataset(self.cfg.dataset, split=self.split, streaming=True)
                    except Exception:
                        # Many Hub repos use train/validation or data/train-*
                        alt = "validation" if self.split == "val" else self.split
                        ds = load_dataset(self.cfg.dataset, split=alt, streaming=True)
                # Shuffle buffer for train
                if self.split == "train":
                    ds = ds.shuffle(seed=self.cfg.seed + epoch, buffer_size=10_000)
                return ds
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to load dataset after retries: {last_err}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        rank = get_rank()
        world = get_world_size()

        epoch = self.cursor.epoch
        buffer: List[int] = []

        while True:
            ds = self._load_stream(epoch)
            row = 0
            skip = self.cursor.row_offset if epoch == self.cursor.epoch else 0

            for ex in ds:
                # Stride across ranks and workers
                if (row + rank) % world != 0:
                    row += 1
                    continue
                local_i = (row // world)
                if local_i % num_workers != worker_id:
                    row += 1
                    continue
                if skip > 0:
                    skip -= 1
                    row += 1
                    continue

                text = ex.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    row += 1
                    continue
                source = ex.get("source")
                # Rough char budget ~ 4 chars/token
                for chunk in chunk_long_text(text, max_chars=self.seq_len * 6):
                    ids = self.tok.encode(chunk, add_special_tokens=False)
                    if not ids:
                        continue
                    if (
                        self.split == "train"
                        and self.cfg.fim_rate > 0
                        and _is_code_source(source, self.cfg.code_sources + CODE_SOURCE_HINTS)
                        and self.rng.random() < self.cfg.fim_rate
                    ):
                        ids = apply_fim(ids, self.tok, self.rng)
                    ids = ids + [self.eos_id]
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len:
                        piece = buffer[: self.seq_len]
                        buffer = buffer[self.seq_len :]
                        ids_t = torch.tensor(piece, dtype=torch.long)
                        yield {
                            "input_ids": ids_t,
                            "labels": ids_t.clone(),
                            "cursor_epoch": torch.tensor(epoch, dtype=torch.long),
                            "cursor_row": torch.tensor(row, dtype=torch.long),
                        }
                row += 1

            epoch += 1
            self.cursor = DataCursor(epoch=epoch, shard_idx=0, row_offset=0)


def build_dataloader(
    cfg: TrainConfig,
    tok: Any,
    split: str,
    cursor: Optional[DataCursor] = None,
) -> DataLoader:
    ds = PackedFIMIterable(cfg, tok, split=split, cursor=cursor)
    nw = cfg.num_workers if split == "train" else 0
    kwargs: Dict[str, Any] = {
        "batch_size": cfg.micro_batch_size,
        "num_workers": nw,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _collate,
    }
    if nw > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "cursor_epoch": batch[-1]["cursor_epoch"],
        "cursor_row": batch[-1]["cursor_row"],
    }


# =============================================================================
# 5. Optim, LR, FSDP2, Checkpoint
# =============================================================================


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    seen: set[int] = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue  # tied embeddings / shared weights
        seen.add(pid)
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: Dict[str, Any] = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": 1e-8,
    }
    # fused AdamW is CUDA-only and not on all builds
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    # Cosine to min_lr
    min_lr = cfg.lr * cfg.min_lr_ratio
    # Estimate total steps from token budget
    tokens_per_step = (
        cfg.micro_batch_size
        * cfg.model.max_seq_len
        * cfg.grad_accum_steps
        * get_world_size()
    )
    total_steps = max(cfg.token_budget // max(tokens_per_step, 1), cfg.warmup_steps + 1)
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (cfg.lr - min_lr) * cos


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def apply_fsdp2(model: CoderLM, device: torch.device) -> nn.Module:
    if not is_dist() or not _HAS_FSDP2 or fully_shard is None:
        return model.to(device)

    mesh = init_device_mesh("cuda", (get_world_size(),), mesh_dim_names=("dp",))
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    return model


def sequence_parallel_enabled(cfg: TrainConfig) -> bool:
    """
    Ulysses-style sequence parallel is reserved for Phase B CPT (32K/128K).
    Phase A (8K) relies on FSDP2 data parallel only — SP is a no-op here.
    """
    return cfg.model.max_seq_len >= 32768 and get_world_size() > 1 and is_dist()


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val: float = float("inf")
    tokens_since_best: int = 0
    nan_count_window: int = 0
    steps_in_window: int = 0
    cursor: DataCursor = field(default_factory=DataCursor)
    mtp_lambda: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val": self.best_val,
            "tokens_since_best": self.tokens_since_best,
            "nan_count_window": self.nan_count_window,
            "steps_in_window": self.steps_in_window,
            "cursor": self.cursor.to_dict(),
            "mtp_lambda": self.mtp_lambda,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainState":
        return cls(
            step=int(d.get("step", 0)),
            tokens_seen=int(d.get("tokens_seen", 0)),
            best_val=float(d.get("best_val", float("inf"))),
            tokens_since_best=int(d.get("tokens_since_best", 0)),
            nan_count_window=int(d.get("nan_count_window", 0)),
            steps_in_window=int(d.get("steps_in_window", 0)),
            cursor=DataCursor.from_dict(d.get("cursor", {})),
            mtp_lambda=float(d.get("mtp_lambda", 0.3)),
        )


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    cfg: TrainConfig,
) -> Path:
    ckpt_dir = output_dir / f"step_{state.step:08d}"
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "train_state.json").write_text(json.dumps(state.to_dict(), indent=2))
        (ckpt_dir / "train_config.json").write_text(
            json.dumps({"train": _cfg_public(cfg)}, indent=2, default=str)
        )
    barrier()

    if is_dist() and _HAS_FSDP2 and dcp is not None:
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
        dcp.save(
            {"model": model_sd, "optim": optim_sd},
            checkpoint_id=str(ckpt_dir / "dcp"),
        )
    else:
        if is_main():
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "state": state.to_dict(),
                },
                ckpt_dir / "checkpoint.pt",
            )

    if is_main():
        latest = output_dir / "LATEST"
        latest.write_text(str(ckpt_dir))
        LOG.info("Saved checkpoint %s", ckpt_dir)
    barrier()
    return ckpt_dir


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> TrainState:
    state_file = ckpt_path / "train_state.json"
    if state_file.exists():
        state = TrainState.from_dict(json.loads(state_file.read_text()))
    else:
        state = TrainState()

    if (ckpt_path / "dcp").exists() and is_dist() and _HAS_FSDP2 and dcp is not None:
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
        payload = {"model": model_sd, "optim": optim_sd}
        dcp.load(payload, checkpoint_id=str(ckpt_path / "dcp"))
        set_model_state_dict(model, model_sd, options=options)
        set_optimizer_state_dict(model, optimizer, optim_sd, options=options)
    else:
        pt = ckpt_path / "checkpoint.pt"
        if not pt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optim"])
        if "state" in blob:
            state = TrainState.from_dict(blob["state"])
    LOG.info("Resumed from %s (step=%s tokens=%s)", ckpt_path, state.step, state.tokens_seen)
    return state


def _cfg_public(cfg: TrainConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    return d


# =============================================================================
# 6. Train / val loop
# =============================================================================


def tokens_per_optimizer_step(cfg: TrainConfig) -> int:
    return (
        cfg.micro_batch_size
        * cfg.model.max_seq_len
        * cfg.grad_accum_steps
        * get_world_size()
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    cfg: TrainConfig,
    tok: Any,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    loader = build_dataloader(cfg, tok, split="val")
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
            out = model(ids, labels=labels, mtp_lambda=0.0)
        loss = out["loss"].detach()
        if is_dist():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / get_world_size()
        total_loss += float(loss.item())
        n += 1
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def smoke_generate(model: nn.Module, tok: Any, device: torch.device, prompt: str) -> str:
    if not is_main():
        return ""
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    # Unwrap if needed — call generate on underlying module
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "generate"):
        out = raw.generate(ids, max_new=64)
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    else:
        text = prompt
    model.train()
    return text


def train_loop(cfg: TrainConfig) -> None:
    device = init_distributed(cfg)
    rank = get_rank()
    setup_logging(rank)
    seed_everything(cfg.seed, rank)

    if is_main():
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        LOG.info("flash_attn=%s fsdp2=%s world=%s device=%s", _HAS_FLASH, _HAS_FSDP2, get_world_size(), device)
        LOG.info("sequence_parallel=%s (enabled only when max_seq_len>=32K)", sequence_parallel_enabled(cfg))
        if cfg.pack_cache_dir:
            LOG.info(
                "pack_cache_dir=%s set — online streaming packer is used; "
                "pre-tokenized cache can be pointed via --data_dir to local parquet",
                cfg.pack_cache_dir,
            )
        LOG.info("Config: %s", json.dumps(_cfg_public(cfg), indent=2, default=str))

    tok = load_qwen_tokenizer(cfg.tokenizer_name)
    # Align vocab (pad up to multiple of 8)
    vocab = max(len(tok), getattr(tok, "vocab_size", 0) or 0)
    if hasattr(tok, "vocab_size") and tok.vocab_size:
        vocab = max(vocab, tok.vocab_size)
    padded = ((vocab + 7) // 8) * 8
    cfg.model.vocab_size = padded

    model = CoderLM(cfg.model)
    counts = model.count_parameters()
    if is_main():
        LOG.info(
            "Parameters: total=%.3fB active_est=%.3fB",
            counts["total_b"],
            counts["active_est_b"],
        )

    model.gradient_checkpointing = cfg.gradient_checkpointing
    model = apply_fsdp2(model, device)
    if not is_dist():
        model = model.to(device)

    if cfg.use_compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            if is_main():
                LOG.info("torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            LOG.warning("torch.compile failed: %s", e)

    optimizer = build_optimizer(model, cfg)
    state = TrainState(mtp_lambda=cfg.mtp_lambda)

    resume_path = cfg.resume
    if resume_path is None:
        latest = Path(cfg.output_dir) / "LATEST"
        if latest.exists():
            resume_path = latest.read_text().strip()
    if resume_path:
        state = load_checkpoint(Path(resume_path), model, optimizer)

    train_loader = build_dataloader(cfg, tok, split="train", cursor=state.cursor)
    train_iter = iter(train_loader)

    writer: Optional[SummaryWriter] = None
    if is_main():
        writer = SummaryWriter(log_dir=str(Path(cfg.output_dir) / "tb"))
        if cfg.use_wandb and _wandb is not None:
            _wandb.init(project=cfg.wandb_project, config=_cfg_public(cfg))

    tps = tokens_per_optimizer_step(cfg)
    if is_main():
        LOG.info("Tokens/optimizer step ≈ %s", f"{tps:,}")

    model.train()
    last_ckpt_time = time.time()
    running_loss = 0.0
    step_t0 = time.time()

    try:
        while state.tokens_seen < cfg.token_budget:
            if cfg.max_steps is not None and state.step >= cfg.max_steps:
                break

            # MTP lambda schedule
            frac = state.tokens_seen / max(cfg.token_budget, 1)
            state.mtp_lambda = (
                cfg.mtp_lambda_late if frac >= cfg.mtp_lambda_switch_frac else cfg.mtp_lambda
            )
            lr = lr_at_step(state.step, cfg)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            oom = False

            for micro in range(cfg.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                state.cursor = DataCursor(
                    epoch=int(batch["cursor_epoch"].item()),
                    shard_idx=0,
                    row_offset=int(batch["cursor_row"].item()),
                )

                try:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=cfg.bf16 and device.type == "cuda",
                    ):
                        out = model(ids, labels=labels, mtp_lambda=state.mtp_lambda)
                        loss = out["loss"] / cfg.grad_accum_steps
                    loss.backward()
                    accum_loss += float(out["loss"].detach().item())
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    LOG.error(
                        "CUDA OOM at step %s micro=%s — lower --micro_batch_size or --max_seq_len",
                        state.step,
                        micro,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    break

            if oom:
                raise SystemExit(2)

            # NaN check
            bad = not math.isfinite(accum_loss)
            state.steps_in_window += 1
            if bad:
                state.nan_count_window += 1
                LOG.warning("Non-finite loss at step %s — skipping update", state.step)
                optimizer.zero_grad(set_to_none=True)
                if state.steps_in_window >= 100:
                    if state.nan_count_window >= 3:
                        raise RuntimeError("Too many NaNs in 100-step window — halting")
                    state.nan_count_window = 0
                    state.steps_in_window = 0
                continue

            if state.steps_in_window >= 100:
                state.nan_count_window = 0
                state.steps_in_window = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            state.step += 1
            state.tokens_seen += tps
            running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss else accum_loss

            if state.step % cfg.log_every_steps == 0 and is_main():
                dt = time.time() - step_t0
                tok_s = (cfg.log_every_steps * tps) / max(dt, 1e-6)
                step_t0 = time.time()
                LOG.info(
                    "step=%s tokens=%s loss=%.4f lr=%.2e tok/s=%.0f mtp_λ=%.2f",
                    state.step,
                    f"{state.tokens_seen:,}",
                    running_loss,
                    lr,
                    tok_s,
                    state.mtp_lambda,
                )
                if writer:
                    writer.add_scalar("train/loss", running_loss, state.step)
                    writer.add_scalar("train/lr", lr, state.step)
                    writer.add_scalar("train/tok_per_s", tok_s, state.step)
                    writer.add_scalar("train/tokens_seen", state.tokens_seen, state.step)
                if cfg.use_wandb and _wandb is not None:
                    _wandb.log(
                        {"loss": running_loss, "lr": lr, "tok_s": tok_s, "tokens": state.tokens_seen},
                        step=state.step,
                    )

            # Checkpoint by steps or wall time
            due_time = (time.time() - last_ckpt_time) >= cfg.ckpt_minutes * 60
            if state.step % cfg.ckpt_every_steps == 0 or due_time:
                save_checkpoint(Path(cfg.output_dir), model, optimizer, state, cfg)
                last_ckpt_time = time.time()

            # Validation
            if state.tokens_seen > 0 and state.tokens_seen % cfg.val_every_tokens < tps:
                val = validate(model, cfg, tok, device)
                if is_main():
                    LOG.info("val_loss=%.4f (best=%.4f)", val, state.best_val)
                    if writer:
                        writer.add_scalar("val/loss", val, state.step)
                    try:
                        sample = smoke_generate(model, tok, device, cfg.smoke_prompt)
                        LOG.info("smoke:\n%s", sample[:500])
                    except Exception as e:  # noqa: BLE001
                        LOG.warning("smoke generate failed: %s", e)

                if val < state.best_val:
                    state.best_val = val
                    state.tokens_since_best = 0
                else:
                    state.tokens_since_best += cfg.val_every_tokens

                if state.tokens_since_best >= cfg.early_stop_tokens:
                    if is_main():
                        LOG.info("Early stop: no val improvement for %s tokens", state.tokens_since_best)
                    break

    except KeyboardInterrupt:
        if is_main():
            LOG.info("Interrupted — saving checkpoint")
        save_checkpoint(Path(cfg.output_dir), model, optimizer, state, cfg)
        raise
    except Exception:
        if is_main():
            LOG.error("Fatal error:\n%s", traceback.format_exc())
        raise
    finally:
        save_checkpoint(Path(cfg.output_dir), model, optimizer, state, cfg)
        if writer:
            writer.close()
        if cfg.use_wandb and _wandb is not None and is_main():
            _wandb.finish()
        if is_dist():
            dist.destroy_process_group()

    if is_main():
        LOG.info("Training finished. tokens_seen=%s step=%s", state.tokens_seen, state.step)


# =============================================================================
# 7. CLI
# =============================================================================


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Phase A MLA+MoE coder pretrain")
    p.add_argument("--dataset", type=str, default=TrainConfig.dataset)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=TrainConfig.tokenizer_name)
    p.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--rope_theta", type=float, default=500_000.0)
    p.add_argument("--rope_factor", type=float, default=1.0)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--token_budget", type=int, default=40_000_000_000)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--fim_rate", type=float, default=0.4)
    p.add_argument("--ckpt_every_steps", type=int, default=1000)
    p.add_argument("--ckpt_minutes", type=float, default=30.0)
    p.add_argument("--val_every_tokens", type=int, default=1_000_000_000)
    p.add_argument("--early_stop_tokens", type=int, default=3_000_000_000)
    p.add_argument("--log_every_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", dest="use_compile", action="store_true")
    p.add_argument("--wandb", dest="use_wandb", action="store_true")
    p.add_argument("--no_distributed", action="store_true")
    p.add_argument("--pack_cache_dir", type=str, default=None)
    p.add_argument("--n_layers", type=int, default=28)
    p.add_argument("--d_model", type=int, default=2048)
    args = p.parse_args(argv)

    model = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        original_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
    )
    return TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        resume=args.resume,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        token_budget=args.token_budget,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        fim_rate=args.fim_rate,
        ckpt_every_steps=args.ckpt_every_steps,
        ckpt_minutes=args.ckpt_minutes,
        val_every_tokens=args.val_every_tokens,
        early_stop_tokens=args.early_stop_tokens,
        log_every_steps=args.log_every_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_compile=args.use_compile,
        use_wandb=args.use_wandb,
        no_distributed=args.no_distributed,
        pack_cache_dir=args.pack_cache_dir,
        model=model,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    train_loop(cfg)


if __name__ == "__main__":
    main()
