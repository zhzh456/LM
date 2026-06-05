"""Collect and save pre-softmax attention scores during eval (per sample / layer / head)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _iter_text_attention_modules(model: nn.Module):
    from patch_sparse_attn import _iter_text_attention_modules as _iter

    return _iter(model)


def enable_attn_score_dump(
    model: nn.Module,
    dump_dir: str,
    *,
    decode_only: bool = True,
) -> None:
    """Attach a shared buffer on all language self-attn modules."""
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    model._attn_dump_dir = str(path)
    model._attn_dump_buffer: List[Tuple[int, torch.Tensor]] = []
    model._attn_dump_decode_only = decode_only
    model._attn_dump_decode_fwd = 0
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        attn._attn_dump_buffer = model._attn_dump_buffer
        attn._attn_dump_root_model = model
        attn._attn_dump_layer_idx = layer_idx
        attn._attn_dump_decode_only = decode_only


def begin_attn_score_sample(model: nn.Module, sample_idx: int) -> None:
    model._attn_dump_buffer = []
    model._attn_dump_sample_idx = sample_idx
    model._attn_dump_decode_fwd = 0
    for attn in _iter_text_attention_modules(model):
        attn._attn_dump_buffer = model._attn_dump_buffer
        attn._attn_dump_root_model = model


def record_pre_softmax_scores(attn_module: nn.Module, scores: torch.Tensor) -> None:
    """
    scores: (batch, num_heads, q_len, kv_len) before softmax.

    decode_only=True (default): only the first decode step (q_len==1),
    i.e. the new token's logits over all prior keys.
    """
    buf = getattr(attn_module, "_attn_dump_buffer", None)
    if buf is None:
        return
    q_len = scores.size(-2)
    decode_only = getattr(attn_module, "_attn_dump_decode_only", True)
    if decode_only:
        if q_len != 1:
            return
        root = getattr(attn_module, "_attn_dump_root_model", None)
        layer_idx = getattr(attn_module, "_attn_dump_layer_idx", 0)
        if root is not None and layer_idx == 0:
            root._attn_dump_decode_fwd = getattr(root, "_attn_dump_decode_fwd", 0) + 1
        fwd = getattr(root, "_attn_dump_decode_fwd", 1) if root is not None else 1
        if fwd != 1:
            return
    layer_idx = getattr(attn_module, "_attn_dump_layer_idx", 0)
    buf.append((layer_idx, scores.detach().float().cpu().half()))


def flush_attn_score_sample(model: nn.Module) -> Optional[Path]:
    dump_dir = getattr(model, "_attn_dump_dir", None)
    buffer: List[Tuple[int, torch.Tensor]] = getattr(model, "_attn_dump_buffer", [])
    sample_idx = getattr(model, "_attn_dump_sample_idx", 0)
    if not dump_dir or not buffer:
        return None

    out_dir = Path(dump_dir) / f"sample_{sample_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, scores in buffer:
        # (batch, heads, q, kv) -> per-head; decode: q=1 -> (kv,) vector
        if scores.dim() == 4:
            scores = scores[0]
        for head_idx in range(scores.size(0)):
            vec = scores[head_idx].squeeze()
            torch.save(vec, out_dir / f"layer_{layer_idx:02d}_head_{head_idx:02d}.pt")
    buffer.clear()
    return out_dir


def patch_baseline_attn_dump(model: nn.Module) -> None:
    """Patch text attention to record standard QK pre-softmax logits (eager path)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if getattr(attn, "_sparse_forward_patched", False):
            continue
        if getattr(attn, "_baseline_attn_dump_patched", False):
            continue

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            record_pre_softmax_scores(self, attn_scores)

            attn_weights = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
            dropout_p = 0.0 if not self.training else self.attention_dropout
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        attn.forward = forward.__get__(attn, type(attn))
        attn._baseline_attn_dump_patched = True
        attn._attn_dump_layer_idx = layer_idx
