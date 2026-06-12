"""Sparse causal attention with per-head relative-position score bias (16K buckets)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# rel_pos[d] = f(d): multiplicative factor on pre-RoPE Q·K/sqrt(d) when (query_pos - key_pos) == d
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

    Index d is f(d): multiplicative factor on pre-RoPE Q·K/sqrt(d) when
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
    """Causal lower triangle (k <= q) using absolute key/query positions.

    Prefill/training (q_len == kv_len): q_idx = 0..q_len-1.
    Decode (q_len == 1, kv_len > 1): q_idx = kv_len-1 so all prior keys are visible.
    """
    q_idx = (kv_len - q_len) + torch.arange(q_len, device=device)
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
        if am.size(-2) > q_len:
            am = am[..., :q_len, :]
        if am.size(-1) < kv_len:
            # Chunked prefill: HF mask covers the current step only; cached keys are valid.
            am = F.pad(am, (kv_len - am.size(-1), 0), value=0)
        elif am.size(-1) > kv_len:
            am = am[..., :kv_len]
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
        visible = visible & _lower_triangular_pair_mask(batch_size, n_heads, q_len, kv_len, device)
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
    Student pre-softmax logits: f(d) × (Q_pre-RoPE · K_pre-RoPE) / sqrt(d) + mask.

    query/key must be pre-RoPE (caller applies RoPE separately for teacher / dense paths).
    Only rel_pos_bias is trainable; Q/K projections stay frozen (qk detached).
    Masked pairs are zeroed before multiply, then additive attention mask.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    n_heads = rel_pos_bias.size(0)
    pos = build_relative_position_bias(rel_pos_bias, q_len, kv_len).to(device=query.device, dtype=dtype)
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
    """Sparse attention: f(d) × (pre-RoPE Q·K / sqrt(d)) at each (p, j)."""
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

    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    distill = None
    if return_distill_tensors:
        # Only scores are needed for MSE distill; avoid keeping duplicate weight tensors.
        distill = {"attn_scores": attn_scores}

    return attn_output, attn_weights, distill


def sparse_scores_only_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    *,
    rel_pos_bias: torch.Tensor,
) -> Tuple[torch.Tensor, None, dict]:
    """
    Distill-only training: compute pre-softmax scores only.

    Skips softmax @ V (loss does not depend on attention output). Saves large
    (B, H, Q, K) weight tensors and their backward buffers at long seq_len.
    """
    batch_size = query.size(0)
    q_len = query.size(-2)
    kv_len = key.size(-2)

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

    b, h, q, d = query.shape
    dummy = torch.zeros(b, q, h * d, device=query.device, dtype=query.dtype)
    return dummy, None, {"attn_scores": attn_scores}


def _loss_dtype(ref: torch.Tensor) -> torch.dtype:
    """Keep distill math in 16-bit when activations are bf16/fp16."""
    if ref.dtype in (torch.bfloat16, torch.float16):
        return ref.dtype
    return torch.bfloat16


def _zero_loss_anchor(t: torch.Tensor) -> torch.Tensor:
    """Scalar 0 tied to ``t`` for autograd; safe when ``t`` has -inf masked slots."""
    return (t.nan_to_num(0.0) * 0.0).sum()


def _kl_per_query_rows(
    full_p: torch.Tensor,
    sparse_q: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    fp = full_p.to(dtype).clamp(min=eps)
    sq = sparse_q.to(dtype).clamp(min=eps)
    kl = (fp * (fp.log() - sq.log())).sum(dim=-1)
    return kl.masked_fill(~valid_rows, 0.0)


class _ChunkedMaskedKL(torch.autograd.Function):
    """KL(full||sparse) with per-query-chunk backward (bf16 softmax, bounded peak memory)."""

    @staticmethod
    def forward(
        ctx,
        sparse_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        valid: torch.Tensor,
        support_mask: torch.Tensor | None,
        query_chunk_size: int,
        kl_dtype: torch.dtype,
    ) -> torch.Tensor:
        neg_inf = torch.tensor(-1e4, device=sparse_scores.device, dtype=kl_dtype)
        eps = 1e-7 if kl_dtype == torch.bfloat16 else 1e-10
        kl_sum = sparse_scores.new_zeros((), dtype=kl_dtype)
        n_rows = sparse_scores.new_zeros((), dtype=kl_dtype)
        chunk = max(1, int(query_chunk_size))
        q_len = sparse_scores.size(2)
        with torch.no_grad():
            for q0 in range(0, q_len, chunk):
                q1 = min(q0 + chunk, q_len)
                valid_chunk = valid[..., q0:q1, :]
                if support_mask is not None:
                    valid_chunk = valid_chunk & support_mask[..., q0:q1, :].bool()
                if not valid_chunk.any():
                    continue
                sparse_logits = sparse_scores[..., q0:q1, :].to(dtype=kl_dtype).masked_fill(~valid_chunk, neg_inf)
                sparse_q = F.softmax(sparse_logits, dim=-1)
                teacher_logits = teacher_scores[..., q0:q1, :].to(dtype=kl_dtype).masked_fill(~valid_chunk, neg_inf)
                full_p = F.softmax(teacher_logits, dim=-1)
                valid_rows = valid_chunk.any(dim=-1)
                if not valid_rows.any():
                    continue
                kl_per_query = _kl_per_query_rows(full_p, sparse_q, valid_rows, eps=eps, dtype=kl_dtype)
                kl_sum = kl_sum + kl_per_query.sum()
                n_rows = n_rows + valid_rows.to(dtype=kl_dtype).sum()
        if n_rows.item() < 0.5:
            ctx.skip_backward = True
            return _zero_loss_anchor(sparse_scores)
        ctx.skip_backward = False
        ctx.query_chunk_size = chunk
        ctx.n_rows = n_rows.clamp(min=1.0)
        ctx.kl_dtype = kl_dtype
        ctx.eps = eps
        ctx.save_for_backward(sparse_scores, teacher_scores, valid, support_mask)
        return kl_sum / ctx.n_rows + _zero_loss_anchor(sparse_scores)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if getattr(ctx, "skip_backward", False):
            return None, None, None, None, None, None
        sparse_scores, teacher_scores, valid, support_mask = ctx.saved_tensors
        grad_scores = torch.zeros_like(sparse_scores)
        chunk = ctx.query_chunk_size
        kl_dtype = ctx.kl_dtype
        n_rows = ctx.n_rows
        neg_inf = torch.tensor(-1e4, device=sparse_scores.device, dtype=kl_dtype)
        eps = ctx.eps
        q_len = sparse_scores.size(2)
        for q0 in range(0, q_len, chunk):
            q1 = min(q0 + chunk, q_len)
            valid_chunk = valid[..., q0:q1, :]
            if support_mask is not None:
                valid_chunk = valid_chunk & support_mask[..., q0:q1, :].bool()
            if not valid_chunk.any():
                continue
            s_leaf = sparse_scores[..., q0:q1, :].detach().requires_grad_(True)
            with torch.enable_grad():
                sparse_logits = s_leaf.to(dtype=kl_dtype).masked_fill(~valid_chunk, neg_inf)
                sparse_q = F.softmax(sparse_logits, dim=-1)
                with torch.no_grad():
                    teacher_logits = teacher_scores[..., q0:q1, :].to(dtype=kl_dtype).masked_fill(~valid_chunk, neg_inf)
                    full_p = F.softmax(teacher_logits, dim=-1)
                valid_rows = valid_chunk.any(dim=-1)
                if not valid_rows.any():
                    continue
                kl_per_query = _kl_per_query_rows(full_p, sparse_q, valid_rows, eps=eps, dtype=kl_dtype)
                chunk_loss = kl_per_query.sum() / n_rows
                g, = torch.autograd.grad(chunk_loss, s_leaf, retain_graph=False)
            grad_scores[..., q0:q1, :] = g
        return grad_output * grad_scores, None, None, None, None, None


def attention_distillation_loss(
    full_weights: torch.Tensor | None = None,
    sparse_weights: torch.Tensor | None = None,
    *,
    sparse_scores: torch.Tensor | None = None,
    teacher_scores: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    prior_key_only: bool = True,
    query_chunk_size: int = 32,
) -> torch.Tensor:
    """KL(full||sparse); chunked backward when ``sparse_scores`` + ``teacher_scores`` are given."""
    if sparse_scores is not None and teacher_scores is not None:
        ref = sparse_scores
        bsz, n_heads, q_len, kv_len = ref.shape
        valid = _build_distill_pair_mask(
            attention_mask,
            batch_size=bsz,
            n_heads=n_heads,
            q_len=q_len,
            kv_len=kv_len,
            device=ref.device,
        )
        if not valid.any():
            return ref.sum() * 0.0
        kl_dtype = _loss_dtype(ref)
        return _ChunkedMaskedKL.apply(
            sparse_scores,
            teacher_scores.detach(),
            valid,
            support_mask,
            int(query_chunk_size),
            kl_dtype,
        )

    if full_weights is None or sparse_weights is None:
        raise ValueError("attention_distillation_loss needs scores pair or weight pair")
    q_len = full_weights.size(-2)
    total = full_weights.new_zeros(())
    count = 0
    for p in range(q_len):
        if p == 0:
            continue
        num_keys = p if prior_key_only else p + 1
        row_dtype = _loss_dtype(full_weights)
        full_row = full_weights[..., p, :num_keys].to(dtype=row_dtype)
        sparse_row = sparse_weights[..., p, :num_keys].to(dtype=row_dtype)
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
    """Teacher pre-softmax logits: RoPE(Q)·RoPE(K) / sqrt(d) + mask (same mask path as student)."""
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
        dtype=torch.bfloat16,
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


class _ChunkedScoresMSELoss(torch.autograd.Function):
    """MSE on score maps with per-query-chunk backward (caps softmax/MSE grad peak memory)."""

    @staticmethod
    def forward(
        ctx,
        student: torch.Tensor,
        teacher: torch.Tensor,
        valid: torch.Tensor,
        query_chunk_size: int,
    ) -> torch.Tensor:
        q_len = student.size(2)
        chunk = max(1, int(query_chunk_size))
        acc_dtype = _loss_dtype(student)
        total_sse = student.new_zeros((), dtype=acc_dtype)
        total_count = student.new_zeros((), dtype=acc_dtype)
        with torch.no_grad():
            for q0 in range(0, q_len, chunk):
                q1 = min(q0 + chunk, q_len)
                v = valid[:, :, q0:q1, :]
                if not v.any():
                    continue
                s = student[:, :, q0:q1, :]
                t = teacher[:, :, q0:q1, :]
                diff = torch.where(v, s - t, torch.zeros_like(s))
                total_sse = total_sse + diff.pow(2).sum()
                total_count = total_count + v.to(dtype=acc_dtype).sum()
        if total_count.item() < 0.5:
            ctx.skip_backward = True
            return _zero_loss_anchor(student)
        ctx.skip_backward = False
        ctx.query_chunk_size = chunk
        ctx.total_count = total_count.clamp(min=1.0)
        ctx.acc_dtype = acc_dtype
        ctx.save_for_backward(student, teacher, valid)
        loss = total_sse / ctx.total_count
        return loss + _zero_loss_anchor(student)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if getattr(ctx, "skip_backward", False):
            return None, None, None, None
        student, teacher, valid = ctx.saved_tensors
        grad_student = torch.zeros_like(student)
        chunk = ctx.query_chunk_size
        inv_denom = (1.0 / ctx.total_count).to(dtype=ctx.acc_dtype)
        q_len = student.size(2)
        for q0 in range(0, q_len, chunk):
            q1 = min(q0 + chunk, q_len)
            v = valid[:, :, q0:q1, :]
            if not v.any():
                continue
            s = student[:, :, q0:q1, :]
            t = teacher[:, :, q0:q1, :]
            diff = torch.where(v, s - t, torch.zeros_like(s))
            grad_student[:, :, q0:q1, :] = 2.0 * diff * inv_denom
        return grad_output * grad_student, None, None, None


def attention_scores_distillation_loss(
    full_scores: torch.Tensor,
    sparse_scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    query_chunk_size: int = 32,
) -> torch.Tensor:
    """
    MSE on pre-softmax logits (student: f(d)×pre-RoPE QK vs teacher: RoPE QK).

    Valid pairs match forward: causal lower-triangle (k <= q) ∩ attention_mask.
    Backward is query-chunked in bf16/fp16 to avoid a single large fp32 grad buffer at seq~6K.
    """
    teacher = full_scores.detach()
    student = sparse_scores
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
        return _zero_loss_anchor(student)

    return _ChunkedScoresMSELoss.apply(student, teacher, valid, int(query_chunk_size))


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

    attn_weights = F.softmax(attn_scores, dim=-1)
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
    """Eval/decode sparse path: same student formula as training (pre-RoPE Q·K)."""
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

    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
