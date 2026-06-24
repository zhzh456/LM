"""Post-RoPE attention: keep top ceil(ratio * n_valid) keys by Q·K logits."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_attention import (
    _forward_attend_pair_mask,
    full_qk_eager_attention_forward,
    qk_pre_softmax_scores,
)


def _get_language_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    lm = getattr(model, "model", model)
    if hasattr(lm, "language_model"):
        return lm.language_model
    if hasattr(lm, "layers"):
        return lm
    raise AttributeError("Cannot find language model on Qwen3-VL model")


def iter_text_attention_modules(model: nn.Module) -> Iterable[nn.Module]:
    for layer in _get_language_model(model).layers:
        yield layer.self_attn


def apply_post_rope_topk_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """
    Mask post-RoPE logits: keep top ceil(ratio * n_valid) keys per (B, H, Q).

    Ranking uses raw attention logits (after additive mask), not softmax probs.
    ratio >= 1.0 returns scores unchanged (caller still applies valid mask).
    """
    if ratio >= 1.0:
        return scores

    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~valid, neg_inf)
    n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    k_keep = torch.ceil(n_valid.float() * float(ratio)).long().clamp(min=1)
    max_k = int(k_keep.max().item())
    _, idx = torch.topk(masked, max_k, dim=-1)
    ranks = torch.arange(max_k, device=scores.device).view(1, 1, 1, -1)
    in_topk = ranks < k_keep
    keep = torch.zeros_like(valid)
    keep.scatter_(dim=-1, index=idx, src=in_topk)
    keep = keep & valid
    return scores.masked_fill(~keep, neg_inf)


def post_rope_topk_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    *,
    topk_ratio: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard post-RoPE Q·K attention with optional top-k key selection."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    batch_size = query.size(0)
    n_heads = query.size(1)
    q_len = query.size(-2)
    kv_len = key.size(-2)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups).to(query.device)

    # qk_pre_softmax_scores applies repeat_kv internally; pass unexpanded key.
    attn_scores = qk_pre_softmax_scores(module, query, key, attention_mask, scaling)
    valid = _forward_attend_pair_mask(
        attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        q_len=q_len,
        kv_len=kv_len,
        device=attn_scores.device,
    )

    if topk_ratio < 1.0:
        attn_scores = apply_post_rope_topk_mask(attn_scores, valid, topk_ratio)
    else:
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~valid, neg_inf)

    attn_weights = F.softmax(attn_scores.float(), dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _patch_attention_forward(attn: nn.Module, ratio: float) -> None:
    if getattr(attn, "_post_rope_topk_patched", False):
        attn._post_rope_topk_ratio = float(ratio)
        return

    orig_forward = attn.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        ratio = float(getattr(self, "_post_rope_topk_ratio", 1.0))
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        dropout = 0.0 if not self.training else self.attention_dropout
        # Prefill only: post-RoPE top-k. Decode (q_len==1) uses full dense attention.
        if query_states.size(-2) == 1:
            attn_output, _ = full_qk_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                dropout=dropout,
            )
        else:
            attn_output, _ = post_rope_topk_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                topk_ratio=ratio,
                dropout=dropout,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn.forward = forward.__get__(attn, type(attn))
    attn._post_rope_topk_patched = True
    attn._post_rope_topk_ratio = float(ratio)
    attn._post_rope_topk_orig_forward = orig_forward


def _unpatch_attention_forward(attn: nn.Module) -> None:
    if not getattr(attn, "_post_rope_topk_patched", False):
        return
    orig = getattr(attn, "_post_rope_topk_orig_forward", None)
    if orig is not None:
        attn.forward = orig
    for key in ("_post_rope_topk_patched", "_post_rope_topk_ratio", "_post_rope_topk_orig_forward"):
        if hasattr(attn, key):
            delattr(attn, key)


def install_post_rope_topk_layers(
    model: nn.Module,
    layer_ratios: Dict[int, float],
) -> None:
    """Patch listed text layers with per-layer retention ratio (same across heads)."""
    for layer_idx, attn in enumerate(iter_text_attention_modules(model)):
        ratio = layer_ratios.get(layer_idx)
        if ratio is None or ratio >= 1.0:
            _unpatch_attention_forward(attn)
            continue
        _patch_attention_forward(attn, ratio)


def clear_post_rope_topk_layers(model: nn.Module) -> None:
    for attn in iter_text_attention_modules(model):
        _unpatch_attention_forward(attn)


def install_post_rope_topk_uniform_layers(
    model: nn.Module,
    *,
    layer_ids: list[int],
    ratio: float,
) -> None:
    """Apply the same retention ratio to all listed layers (prefill top-k)."""
    if ratio >= 1.0:
        clear_post_rope_topk_layers(model)
        return
    install_post_rope_topk_layers(model, {int(layer_id): float(ratio) for layer_id in layer_ids})


def set_post_rope_topk_single_layer(
    model: nn.Module,
    *,
    layer_id: int,
    ratio: float,
) -> None:
    """Only layer_id uses top-k; all other layers run default attention."""
    layer_ratios: Dict[int, float] = {}
    if ratio < 1.0:
        layer_ratios[int(layer_id)] = float(ratio)
    install_post_rope_topk_layers(model, layer_ratios)


def num_text_layers(model: nn.Module) -> int:
    return len(_get_language_model(model).layers)
