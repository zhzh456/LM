"""Sparse causal attention with per-head relative-position score bias (16K buckets)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# rel_pos_score_bias[d] = additive logit when (query_pos - key_pos) == d
REL_POS_BUCKETS = 16384


def build_relative_position_init(
    num_buckets: int,
    *,
    num_heads: int = 1,
    head_idx: int = 0,
    near_tau: float = 128.0,
    wave_period: float = 32.0,
    wave_amp: float = 0.12,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Init prior: nearby distances get larger bias; farther distances decay with damped oscillation.

    - decay ~ exp(-d / near_tau): closer keys start with stronger logits
    - sin wave * exp(-d / (2*near_tau)): adds alternating bumps that fade at long range
    - per-head phase offset: heads do not share identical patterns
    """
    d = torch.arange(num_buckets, dtype=torch.float32)
    decay = torch.exp(-d / near_tau)
    phase = 2.0 * math.pi * (head_idx / max(num_heads, 1))
    envelope = torch.exp(-d / (2.0 * near_tau))
    wave = wave_amp * torch.sin(2.0 * math.pi * d / wave_period + phase) * envelope
    init = decay + wave
    if noise_std > 0:
        init = init + torch.randn_like(init) * noise_std
    return init


class PerHeadRelativePositionBias(nn.Module):
    """
    Each attention head has its own trainable vector of length num_buckets (default 16384).

    Index d is the bias added to pre-softmax attention logits when
    (q_pos - k_pos) == d (d=0: same position, d=1: one step back, ...).
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = REL_POS_BUCKETS,
        *,
        near_tau: float = 128.0,
        wave_period: float = 32.0,
        wave_amp: float = 0.12,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.head_biases = nn.ParameterList(
            [
                nn.Parameter(
                    build_relative_position_init(
                        num_buckets,
                        num_heads=num_heads,
                        head_idx=h,
                        near_tau=near_tau,
                        wave_period=wave_period,
                        wave_amp=wave_amp,
                        noise_std=noise_std,
                    )
                )
                for h in range(num_heads)
            ]
        )

    def stacked(self) -> torch.Tensor:
        """Stack per-head vectors to (num_heads, num_buckets) for batched lookup."""
        return torch.stack(tuple(self.head_biases), dim=0)

    def __getitem__(self, head_idx: int) -> torch.Tensor:
        return self.head_biases[head_idx]


def topk_count_for_position(p: int, ratio: float = 0.2) -> int:
    """Number of keys to keep for query position p (keys 0..p-1), ceil(20% * p)."""
    if p <= 0:
        return 0
    return max(1, math.ceil(ratio * p))


def build_relative_position_bias(
    rel_pos_bias: torch.Tensor,
    q_len: int,
    kv_len: int,
) -> torch.Tensor:
    """
    Lookup per-head bias from relative distance (q_idx - k_idx).

    rel_pos_bias: (num_heads, num_buckets) — one vector per head
    Returns: (1, num_heads, q_len, kv_len)
    """
    device = rel_pos_bias.device
    # Absolute positions: prefill uses 0..q_len-1; decode (q_len=1, kv_len>1) uses q at kv_len-1.
    q_pos = (kv_len - q_len) + torch.arange(q_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    rel_dist = q_pos[:, None] - k_pos[None, :]  # (q, k); d=0 on diagonal
    max_bucket = rel_pos_bias.shape[1]
    rel_dist_clamped = rel_dist.clamp(0, max_bucket - 1)
    bias = rel_pos_bias[:, rel_dist_clamped]  # (heads, q, k)
    bias = bias.masked_fill((rel_dist < 0).unsqueeze(0), 0.0)
    return bias.unsqueeze(0)


def apply_sparse_topk_mask(
    scores: torch.Tensor,
    topk_ratio: float = 0.2,
) -> torch.Tensor:
    """Deprecated: kept for compatibility, now returns input unchanged."""
    return scores


def _lower_triangular_pair_mask(
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Causal lower triangle (k <= q), same rule as HF causal_mask_function."""
    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(kv_len, device=device)
    causal = k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)
    return causal.view(1, 1, q_len, kv_len).expand(batch_size, n_heads, q_len, kv_len)


def _causal_additive_attention_mask(
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask: 0 on lower triangle (incl. diagonal), finfo.min on upper."""
    allowed = _lower_triangular_pair_mask(batch_size, n_heads, q_len, kv_len, device)
    blocked = torch.finfo(dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=device, dtype=dtype),
        blocked,
    )


def _prepare_additive_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Expand HF additive mask to (B, H, Q, K) on target device/dtype."""
    if attention_mask is None:
        return None

    am = attention_mask.to(device=device)
    if am.dim() == 4:
        if am.size(-2) != q_len or am.size(-1) != kv_len:
            am = am[..., :q_len, :kv_len]
        if am.size(0) == 1 and batch_size > 1:
            am = am.expand(batch_size, -1, -1, -1)
        if am.size(1) == 1:
            am = am.expand(-1, n_heads, -1, -1)
        elif am.size(1) != n_heads:
            am = am[:, :1].expand(batch_size, n_heads, q_len, kv_len)
        return am.to(dtype=dtype)

    if am.dim() == 2:
        if am.size(-1) < kv_len:
            am = F.pad(am, (0, kv_len - am.size(-1)), value=0)
        elif am.size(-1) > kv_len:
            am = am[..., :kv_len]
        real = am.bool()
        q_ok = real.view(batch_size, 1, q_len, 1)
        k_ok = real.view(batch_size, 1, 1, kv_len)
        visible = q_ok & k_ok
        visible = visible & _lower_triangular_pair_mask(
            batch_size, n_heads, q_len, kv_len, device
        )
        blocked = torch.finfo(dtype).min
        return torch.where(
            visible,
            torch.zeros((), device=device, dtype=dtype),
            blocked,
        )

    return am.to(dtype=dtype)


def _pair_attend_gate(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Float gate (B, H, Q, K): 1.0 where attention_mask allows attending, else 0."""
    prepared = _prepare_additive_attention_mask(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype=dtype,
    )
    if prepared is None:
        return None
    return (prepared > -1e4).to(dtype=dtype)


def _apply_pre_softmax_attention_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
) -> torch.Tensor:
    """Add expanded causal/padding additive mask to pre-softmax logits."""
    prepared = _prepare_additive_attention_mask(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=scores.device,
        dtype=scores.dtype,
    )
    if prepared is None:
        return scores + _causal_additive_attention_mask(
            batch_size,
            n_heads,
            q_len,
            kv_len,
            device=scores.device,
            dtype=scores.dtype,
        )
    return scores + prepared


def _sparse_pre_softmax_scores(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    rel_pos_bias: torch.Tensor,
    batch_size: int,
    q_len: int,
    kv_len: int,
    scaling: float,
    *,
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Pre-softmax logits: rel_pos[d] × (Q·K) + attention_mask.

    Same distance d can map to different logits because qk[p,j] varies per pair.
    Only rel_pos_bias is trainable; Q/K projections stay frozen (qk detached).
    Masked / blocked (p, j) pairs are zeroed before multiply, then additive mask.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    n_heads = rel_pos_bias.size(0)
    pos = build_relative_position_bias(rel_pos_bias, q_len, kv_len).to(dtype)
    pos = pos.to(rel_pos_bias.device)
    if batch_size > 1:
        pos = pos.expand(batch_size, -1, -1, -1).contiguous()

    key_states = repeat_kv(key, module.num_key_value_groups)
    qk = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    qk = qk.detach().to(dtype)

    gate = _pair_attend_gate(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=qk.device,
        dtype=dtype,
    )
    if gate is not None:
        pos = pos * gate
        qk = qk * gate

    scores = pos * qk
    return _apply_pre_softmax_attention_mask(
        scores,
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
    )


def sparse_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    rel_pos_bias: torch.Tensor,
    topk_ratio: float = 0.2,
    return_distill_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
    """Sparse attention: (rel-pos template per d) × (frozen Q·K) at each (p, j)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    batch_size = query.size(0)
    q_len = query.size(-2)
    kv_len = key.size(-2)

    value_states = repeat_kv(value, module.num_key_value_groups).to(query.device)

    attn_scores = _sparse_pre_softmax_scores(
        module,
        query,
        key,
        rel_pos_bias,
        batch_size,
        q_len,
        kv_len,
        scaling,
        attention_mask=attention_mask,
        dtype=query.dtype,
    )

    full_weights = F.softmax(attn_scores.float(), dim=-1).to(query.dtype)
    attn_weights = F.dropout(full_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    distill = None
    if return_distill_tensors:
        distill = {
            "full_weights": full_weights,
            "sparse_weights": attn_weights,
            "attn_scores": attn_scores,
        }

    return attn_output, attn_weights, distill


def attention_distillation_loss(
    full_weights: torch.Tensor,
    sparse_weights: torch.Tensor,
    *,
    prior_key_only: bool = True,
) -> torch.Tensor:
    """KL(full || sparse) on keys j < p for each query position p."""
    q_len = full_weights.size(-2)
    total = full_weights.new_zeros(())
    count = 0

    for p in range(q_len):
        if p == 0:
            continue
        num_keys = p if prior_key_only else p + 1
        full_row = full_weights[..., p, :num_keys].float()
        sparse_row = sparse_weights[..., p, :num_keys].float()
        full_row = full_row / full_row.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sparse_row = sparse_row / sparse_row.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        kl = F.kl_div(sparse_row.log().clamp(min=-1e4), full_row.detach(), reduction="batchmean")
        total = total + kl
        count += 1

    if count == 0:
        return total
    return total / count


def attention_output_distillation_loss(
    full_output: torch.Tensor,
    sparse_output: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(sparse_output, full_output.detach())


def qk_pre_softmax_scores(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Teacher pre-softmax logits: Q·K + attention_mask (same mask path as student)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    batch_size = query.size(0)
    n_heads = query.size(1)
    q_len = query.size(-2)
    kv_len = key.size(-2)

    key_states = repeat_kv(key, module.num_key_value_groups)
    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    gate = _pair_attend_gate(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=attn_scores.device,
        dtype=attn_scores.dtype,
    )
    if gate is not None:
        attn_scores = attn_scores * gate

    return _apply_pre_softmax_attention_mask(
        attn_scores,
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
    )


def _forward_attend_pair_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool (B,H,Q,K): causal lower triangle ∩ HF attention_mask (same as forward)."""
    valid = _lower_triangular_pair_mask(batch_size, n_heads, q_len, kv_len, device)
    gate = _pair_attend_gate(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype=torch.float32,
    )
    if gate is not None:
        valid = valid & gate.bool()
    return valid


def _build_distill_pair_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    return _forward_attend_pair_mask(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
    )


def attention_scores_distillation_loss(
    full_scores: torch.Tensor,
    sparse_scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MSE on pre-softmax logits (sparse: rel_pos[d]×QK vs teacher QK).

    Valid pairs match forward: causal lower-triangle (k <= q) ∩ attention_mask,
    same visible region as full (teacher) and sparse (student) pre-softmax paths.
    """
    teacher = full_scores.detach().float()
    student = sparse_scores.float()
    bsz, n_heads, q_len, kv_len = teacher.shape

    valid = _build_distill_pair_mask(
        attention_mask,
        batch_size=bsz,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=teacher.device,
    )
    valid = valid & torch.isfinite(teacher) & torch.isfinite(student)
    if not valid.any():
        return student.sum() * 0.0
    return F.mse_loss(student[valid], teacher[valid])


def full_qk_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard QK attention (prefill / eval baseline path)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups).to(query.device)

    attn_scores = qk_pre_softmax_scores(module, query, key, attention_mask, scaling)

    try:
        from attn_score_dump import record_pre_softmax_scores

        record_pre_softmax_scores(module, attn_scores)
    except ImportError:
        pass

    attn_weights = F.softmax(attn_scores.float(), dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def dense_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    rel_pos_bias: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eval/decode sparse path: same score formula as training (pos[d] × frozen Q·K)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    batch_size = query.size(0)
    q_len = query.size(-2)
    kv_len = key.size(-2)

    value_states = repeat_kv(value, module.num_key_value_groups).to(query.device)

    if rel_pos_bias is None:
        rel_pos_bias = torch.zeros(
            module.config.num_attention_heads,
            REL_POS_BUCKETS,
            device=query.device,
            dtype=query.dtype,
        )
    attn_scores = _sparse_pre_softmax_scores(
        module,
        query,
        key,
        rel_pos_bias,
        batch_size,
        q_len,
        kv_len,
        scaling,
        attention_mask=attention_mask,
        dtype=query.dtype,
    )

    try:
        from attn_score_dump import record_pre_softmax_scores

        record_pre_softmax_scores(module, attn_scores)
    except ImportError:
        pass

    attn_weights = F.softmax(attn_scores.float(), dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
