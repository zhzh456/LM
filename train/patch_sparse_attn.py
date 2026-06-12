"""Freeze Qwen3-VL and patch text self-attention with sparse + rel-pos bias modules."""

from __future__ import annotations

import os
from typing import Iterable, List

import torch
import torch.nn as nn

SPARSE_REL_POS_FILENAME = "sparse_rel_pos_bias.pt"
SPARSE_CKPT_META_KEY = "_meta"

from sparse_attention import (
    DEFAULT_CONTENT_TOPK_K,
    DEFAULT_SPARSE_TOPK_K,
    DEFAULT_STE_TAU,
    PerHeadRelativePositionBias,
    attention_distance_gap_recall_loss,
    dense_eager_attention_forward,
    full_qk_eager_attention_forward,
    qk_pre_softmax_scores,
    sparse_union_attention_forward,
    sparse_union_scores_only_forward,
)


def _get_language_model(model: nn.Module) -> nn.Module:
    lm = getattr(model, "model", model)
    if hasattr(lm, "language_model"):
        return lm.language_model
    if hasattr(lm, "layers"):
        return lm
    raise AttributeError("Cannot find language model on Qwen3-VL model")


def _iter_text_attention_modules(model: nn.Module) -> Iterable[nn.Module]:
    for layer in _get_language_model(model).layers:
        yield layer.self_attn


def _patch_text_model_early_stop(text_model: nn.Module, stop_at_layer_id: int) -> None:
    """Run text decoder layers 0..stop_at_layer_id only (skip later layers + final norm is cheap)."""
    text_model._sparse_stop_at_layer_id = int(stop_at_layer_id)
    if getattr(text_model, "_sparse_early_stop_patched", False):
        return

    _orig_forward = text_model.forward

    def forward(self, *args, **kwargs):
        stop_at = getattr(self, "_sparse_stop_at_layer_id", None)
        if stop_at is None:
            return _orig_forward(*args, **kwargs)

        all_layers = self.layers
        stop_at = min(int(stop_at), len(all_layers) - 1)
        self.layers = nn.ModuleList(list(all_layers)[: stop_at + 1])
        try:
            return _orig_forward(*args, **kwargs)
        finally:
            self.layers = all_layers

    text_model.forward = forward.__get__(text_model, type(text_model))
    text_model._sparse_early_stop_patched = True


def _patch_causal_lm_sparse_forward(model: nn.Module) -> None:
    """Skip lm_head when sparse training only needs early-stopped backbone."""
    if getattr(model, "_sparse_causal_lm_patched", False):
        return

    _orig_forward = model.forward

    def forward(self, *args, labels=None, logits_to_keep=0, **kwargs):
        text_model = _get_language_model(self)
        if getattr(text_model, "_sparse_stop_at_layer_id", None) is None:
            return _orig_forward(*args, labels=labels, logits_to_keep=logits_to_keep, **kwargs)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLCausalLMOutputWithPast,
        )

        model_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        outputs = self.model(**model_kwargs)
        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    model.forward = forward.__get__(model, type(model))
    model._sparse_causal_lm_patched = True


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def attach_sparse_attention_modules(
    model: nn.Module,
    *,
    layer_id: int = 35,
    num_buckets: int = 16384,
    near_tau: float = 128.0,
    wave_period: float = 32.0,
    wave_amp: float = 0.12,
    noise_std: float = 0.01,
    sparse_topk_k: int = DEFAULT_SPARSE_TOPK_K,
    content_topk_k: int = DEFAULT_CONTENT_TOPK_K,
    target_topk_k: int | None = None,
    ste_tau: float = DEFAULT_STE_TAU,
    sparse_gap_recall_weight: float = 1.0,
    sparse_dist_score_scale: float = 0.75,
    distance_ste_local: bool = True,
) -> List[nn.Parameter]:
    """Register PerHeadRelativePositionBias (S[d]) on one text self-attn layer."""
    trainable: List[nn.Parameter] = []
    n_layers = 0
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        n_layers += 1
        if layer_idx != layer_id:
            continue
        n_heads = attn.config.num_attention_heads
        bias_module = PerHeadRelativePositionBias(
            n_heads,
            num_buckets=num_buckets,
            near_tau=near_tau,
            wave_period=wave_period,
            wave_amp=wave_amp,
            noise_std=noise_std,
        )
        layer_device = next(attn.parameters()).device
        bias_module = bias_module.to(device=layer_device, dtype=torch.float32)
        attn.add_module("rel_pos_bias_per_head", bias_module)
        attn.sparse_content_topk_k = int(content_topk_k)
        attn.sparse_distance_topk_k = int(sparse_topk_k)
        attn.sparse_target_topk_k = int(target_topk_k if target_topk_k is not None else sparse_topk_k)
        attn.sparse_ste_tau = float(ste_tau)
        attn.sparse_gap_recall_weight = float(sparse_gap_recall_weight)
        attn.sparse_dist_score_scale = float(sparse_dist_score_scale)
        attn.sparse_distance_ste_local = bool(distance_ste_local)
        attn.sparse_topk_ratio = 0.2
        attn.use_sparse_attention = True
        attn._sparse_distill_loss = None
        attn._sparse_layer_idx = layer_idx
        for head_idx, p in enumerate(bias_module.head_biases):
            p.requires_grad = True
            trainable.append(p)
        return trainable
    raise ValueError(f"layer_id={layer_id} out of range (num_text_layers={n_layers})")


def _get_stacked_rel_pos_bias(attn: nn.Module) -> torch.Tensor:
    return attn.rel_pos_bias_per_head.stacked()


def reset_sparse_pre_rope_key_caches(model: nn.Module) -> None:
    """Clear per-layer pre-RoPE K cache before a new generate() / sample."""
    for attn in _iter_text_attention_modules(model):
        if hasattr(attn, "_pre_rope_key_cache"):
            attn._pre_rope_key_cache = None


def _wrap_generate_reset_pre_rope_cache(model: nn.Module) -> None:
    """Reset pre-RoPE K cache at the start of each model.generate() call."""
    if getattr(model, "_sparse_generate_reset_wrapped", False):
        return

    _orig_generate = model.generate

    def generate(*args, **kwargs):
        reset_sparse_pre_rope_key_caches(model)
        return _orig_generate(*args, **kwargs)

    model.generate = generate
    model._sparse_generate_reset_wrapped = True


def _assemble_pre_rope_keys(
    attn: nn.Module,
    key_pre_rope_step: torch.Tensor,
    *,
    track_cache: bool,
) -> torch.Tensor:
    """
    Build full pre-RoPE K for sparse scores: f(d) * Q_pre * K_pre / sqrt(d).

    Prefill (step has all tokens): use keys from this forward.
    Decode (one new token): concat cached pre-RoPE keys + new key.
    """
    step_len = key_pre_rope_step.size(-2)
    if step_len > 1:
        cached = getattr(attn, "_pre_rope_key_cache", None)
        if cached is not None and track_cache:
            key_pre_rope = torch.cat([cached, key_pre_rope_step], dim=-2)
        else:
            key_pre_rope = key_pre_rope_step
    else:
        cached = getattr(attn, "_pre_rope_key_cache", None)
        if cached is not None:
            key_pre_rope = torch.cat([cached, key_pre_rope_step], dim=-2)
        else:
            key_pre_rope = key_pre_rope_step
    if track_cache:
        attn._pre_rope_key_cache = key_pre_rope.detach()
    return key_pre_rope


def set_run_distill_this_step(model: nn.Module, enabled: bool) -> None:
    """Toggle dense teacher forward on the sparse-patched layer only."""
    for attn in _iter_text_attention_modules(model):
        if getattr(attn, "use_sparse_attention", False):
            attn._run_distill_this_step = enabled


def _patch_attention_forward(attn: nn.Module) -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    if getattr(attn, "_sparse_forward_patched", False):
        return

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

        # Student sparse scores use pre-RoPE Q/K; teacher uses post-RoPE Q/K.
        query_pre_rope = query_states
        key_pre_rope_step = key_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        use_sparse = getattr(self, "use_sparse_attention", False)
        if use_sparse:
            key_pre_rope = _assemble_pre_rope_keys(
                self,
                key_pre_rope_step,
                track_cache=getattr(self, "sparse_decode_only", False),
            )
        else:
            key_pre_rope = key_pre_rope_step

        rel_bias = _get_stacked_rel_pos_bias(self)
        union_kw = dict(
            rel_pos_bias=rel_bias,
            content_topk_k=int(getattr(self, "sparse_content_topk_k", DEFAULT_CONTENT_TOPK_K)),
            distance_topk_k=int(getattr(self, "sparse_distance_topk_k", DEFAULT_SPARSE_TOPK_K)),
            ste_tau=float(getattr(self, "sparse_ste_tau", DEFAULT_STE_TAU)),
            dist_score_scale=float(getattr(self, "sparse_dist_score_scale", 0.75)),
            distance_ste_local=bool(getattr(self, "sparse_distance_ste_local", True)),
        )

        run_distill = getattr(self, "_run_distill_this_step", True)
        sparse_train_layer = getattr(self, "use_sparse_attention", False) and not getattr(self, "sparse_decode_only", False)
        if sparse_train_layer:
            self._sparse_kl_loss = None
            self._sparse_mse_loss = None
            self._sparse_distill_loss = None
            self._sparse_gap_recall_loss = None
            if self.training:
                sparse_out, _, distill = sparse_union_scores_only_forward(
                    self,
                    query_pre_rope,
                    key_pre_rope,
                    query_states,
                    key_states,
                    attention_mask,
                    self.scaling,
                    **union_kw,
                )
            else:
                sparse_out, _, distill = sparse_union_attention_forward(
                    self,
                    query_pre_rope,
                    key_pre_rope,
                    value_states,
                    query_states,
                    key_states,
                    attention_mask,
                    self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    return_distill_tensors=True,
                    **union_kw,
                )
            if run_distill:
                with torch.no_grad():
                    teacher_scores = qk_pre_softmax_scores(
                        self,
                        query_states,
                        key_states,
                        attention_mask,
                        self.scaling,
                    )
                self._sparse_distill_extras = {
                    "sparse_scores": distill["attn_scores"],
                    "teacher_scores": teacher_scores,
                    "union_mask": distill["union_mask"],
                    "mask_c": distill.get("mask_c"),
                    "mask_d": distill.get("mask_d"),
                }
                self._sparse_distill_attention_mask = attention_mask
            else:
                self._sparse_distill_extras = None
                self._sparse_distill_attention_mask = None
            attn_output = sparse_out
        else:
            dropout = 0.0 if not self.training else self.attention_dropout
            sparse_decode_only = getattr(self, "sparse_decode_only", False)
            if sparse_decode_only and query_states.size(-2) == 1:
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
                attn_output, _ = sparse_union_attention_forward(
                    self,
                    query_pre_rope,
                    key_pre_rope,
                    value_states,
                    query_states,
                    key_states,
                    attention_mask,
                    self.scaling,
                    dropout=dropout,
                    **union_kw,
                )[0:2]
            self._sparse_distill_loss = None
            self._sparse_kl_loss = None
            self._sparse_mse_loss = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn.forward = forward.__get__(attn, type(attn))
    attn._sparse_forward_patched = True


def patch_model_for_sparse_training(
    model: nn.Module,
    *,
    layer_id: int = 35,
    **attach_kwargs,
) -> List[nn.Parameter]:
    """Freeze backbone; train sparse rel-pos on one text layer only."""
    freeze_backbone(model)
    trainable = attach_sparse_attention_modules(model, layer_id=layer_id, **attach_kwargs)
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        attn.sparse_decode_only = False
        _patch_attention_forward(attn)
    _patch_text_model_early_stop(_get_language_model(model), layer_id)
    _patch_causal_lm_sparse_forward(model)
    return trainable


def patch_model_for_sparse_eval(
    model: nn.Module,
    *,
    layer_id: int = 35,
    **attach_kwargs,
) -> None:
    """Eval-only: one layer uses sparse pre-RoPE attention; others keep default."""
    attach_sparse_attention_modules(model, layer_id=layer_id, **attach_kwargs)
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        attn.sparse_decode_only = True
        _patch_attention_forward(attn)
    reset_sparse_pre_rope_key_caches(model)
    _wrap_generate_reset_pre_rope_cache(model)


def finalize_sparse_distill_losses(model: nn.Module) -> None:
    """Compute gap-recall loss after full forward (keeps peak memory out of layer forward)."""
    for attn in _iter_text_attention_modules(model):
        if not getattr(attn, "use_sparse_attention", False):
            continue
        extras = getattr(attn, "_sparse_distill_extras", None)
        if extras is None:
            continue
        attention_mask = getattr(attn, "_sparse_distill_attention_mask", None)
        mask_c = extras.get("mask_c")
        mask_d = extras.get("mask_d")
        if mask_c is None or mask_d is None:
            raise RuntimeError("gap-recall distill requires mask_c and mask_d from sparse forward")
        gap_loss, gap_recall, union_recall = attention_distance_gap_recall_loss(
            extras["teacher_scores"],
            mask_d,
            mask_c,
            attention_mask,
            int(getattr(attn, "sparse_target_topk_k", DEFAULT_SPARSE_TOPK_K)),
            content_topk_k=int(getattr(attn, "sparse_content_topk_k", DEFAULT_CONTENT_TOPK_K)),
        )
        gap_w = float(getattr(attn, "sparse_gap_recall_weight", 1.0))
        distill_loss = gap_w * gap_loss
        attn._sparse_kl_loss = None
        attn._sparse_gap_recall_loss = gap_loss
        attn._sparse_gap_recall = gap_recall
        attn._sparse_topk_recall = union_recall
        attn._sparse_mse_loss = None
        attn._sparse_distill_loss = distill_loss
        attn._sparse_distill_extras = None
        attn._sparse_distill_attention_mask = None


def collect_sparse_distill_loss(model: nn.Module) -> torch.Tensor | None:
    parts = collect_sparse_distill_losses(model)
    return parts.get("distill")


def collect_sparse_distill_losses(model: nn.Module) -> dict[str, torch.Tensor | None]:
    kl_total = gap_total = distill_total = None
    gap_recall = union_recall = None
    n = 0
    for attn in _iter_text_attention_modules(model):
        if not getattr(attn, "use_sparse_attention", False):
            continue
        kl = getattr(attn, "_sparse_kl_loss", None)
        gap = getattr(attn, "_sparse_gap_recall_loss", None)
        distill = getattr(attn, "_sparse_distill_loss", None)
        if kl is not None:
            kl_total = kl if kl_total is None else kl_total + kl
        if gap is not None:
            gap_total = gap if gap_total is None else gap_total + gap
        if distill is not None:
            distill_total = distill if distill_total is None else distill_total + distill
            n += 1
        gr = getattr(attn, "_sparse_gap_recall", None)
        ur = getattr(attn, "_sparse_topk_recall", None)
        if gr is not None:
            gap_recall = gr if gap_recall is None else (gap_recall + gr) / 2.0
        if ur is not None:
            union_recall = ur if union_recall is None else (union_recall + ur) / 2.0
    return {
        "kl": kl_total,
        "gap": gap_total,
        "distill": distill_total,
        "gap_recall": gap_recall,
        "union_recall": union_recall,
        "mse": None,
    }


def trainable_sparse_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def iter_attn_with_bias(model: nn.Module) -> Iterable[nn.Module]:
    for attn in _iter_text_attention_modules(model):
        if hasattr(attn, "rel_pos_bias_per_head"):
            yield attn


def _resize_rel_pos_vector(vec: torch.Tensor, target_len: int) -> torch.Tensor:
    vec = vec.float().flatten()
    if vec.numel() == target_len:
        return vec
    out = torch.zeros(target_len, dtype=torch.float32)
    n = min(target_len, vec.numel())
    out[:n] = vec[:n]
    return out


def unpack_sparse_rel_pos_checkpoint(
    raw: dict,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return (meta, tensors). Supports new (_meta) and legacy flat checkpoints."""
    if SPARSE_CKPT_META_KEY in raw:
        meta = dict(raw[SPARSE_CKPT_META_KEY])
        tensors = {k: v for k, v in raw.items() if k != SPARSE_CKPT_META_KEY and isinstance(v, torch.Tensor)}
        return meta, tensors

    tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    meta: dict = {}
    layer_ids = sorted({int(k.split(".")[0].removeprefix("layer_")) for k in tensors if k.startswith("layer_") and ".head_" in k})
    if len(layer_ids) == 1:
        meta["train_layer_id"] = layer_ids[0]
    return meta, tensors


def save_sparse_rel_pos_checkpoint(model: nn.Module, output_dir: str) -> str:
    """Save single-layer rel-pos weights + metadata to sparse_rel_pos_bias.pt."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, SPARSE_REL_POS_FILENAME)
    attns = list(iter_attn_with_bias(model))
    if not attns:
        raise RuntimeError("No sparse rel-pos modules attached; nothing to save.")

    tensors: dict[str, torch.Tensor] = {}
    layer_ids: list[int] = []
    rel_pos_buckets = None
    num_heads = None
    for attn in attns:
        layer_idx = int(getattr(attn, "_sparse_layer_idx", 0))
        layer_ids.append(layer_idx)
        mod = attn.rel_pos_bias_per_head
        rel_pos_buckets = mod.num_buckets
        num_heads = mod.num_heads
        for h, p in enumerate(mod.head_biases):
            tensors[f"layer_{layer_idx}.head_{h}"] = p.detach().cpu()
        tensors[f"layer_{layer_idx}.sparse_topk_ratio"] = torch.tensor(getattr(attn, "sparse_topk_ratio", 0.2))

    meta = {
        "format_version": 3,
        "train_layer_id": layer_ids[0] if len(layer_ids) == 1 else layer_ids,
        "rel_pos_buckets": rel_pos_buckets,
        "num_heads": num_heads,
        "num_sparse_layers": len(layer_ids),
    }
    if attns:
        a0 = attns[0]
        meta["sparse_topk_k"] = int(getattr(a0, "sparse_distance_topk_k", DEFAULT_SPARSE_TOPK_K))
        meta["content_topk_k"] = int(getattr(a0, "sparse_content_topk_k", DEFAULT_CONTENT_TOPK_K))
        meta["ste_tau"] = float(getattr(a0, "sparse_ste_tau", DEFAULT_STE_TAU))
    payload = {SPARSE_CKPT_META_KEY: meta, **tensors}
    torch.save(payload, path)

    layer_tag = meta["train_layer_id"]
    alias = os.path.join(output_dir, f"sparse_rel_pos_layer{layer_tag}.pt")
    if alias != path:
        torch.save(payload, alias)

    print(
        f"[save] sparse rel-pos layer={layer_tag} heads={num_heads} " f"buckets={rel_pos_buckets} -> {path}",
        flush=True,
    )
    return path


def load_sparse_rel_pos_checkpoint(
    model: nn.Module,
    path: str,
    *,
    expected_layer_id: int | None = None,
) -> int:
    """Load layer_i.head_h tensors; optionally verify train_layer_id matches."""
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid sparse checkpoint (not a dict): {path}")
    meta, tensors = unpack_sparse_rel_pos_checkpoint(raw)

    saved_layer = meta.get("train_layer_id")
    if saved_layer is not None and expected_layer_id is not None:
        if isinstance(saved_layer, list):
            if expected_layer_id not in saved_layer:
                raise ValueError(f"Checkpoint train_layer_id={saved_layer} does not contain " f"expected layer_id={expected_layer_id} ({path})")
        elif int(saved_layer) != int(expected_layer_id):
            raise ValueError(f"Checkpoint train_layer_id={saved_layer} != " f"expected layer_id={expected_layer_id} ({path})")

    loaded = 0
    for attn in iter_attn_with_bias(model):
        layer_idx = int(getattr(attn, "_sparse_layer_idx", 0))
        if expected_layer_id is not None and layer_idx != expected_layer_id:
            continue
        if meta.get("sparse_topk_k") is not None:
            attn.sparse_distance_topk_k = int(meta["sparse_topk_k"])
        if meta.get("content_topk_k") is not None:
            attn.sparse_content_topk_k = int(meta["content_topk_k"])
        if meta.get("ste_tau") is not None:
            attn.sparse_ste_tau = float(meta["ste_tau"])
        for h, p in enumerate(attn.rel_pos_bias_per_head.head_biases):
            key = f"layer_{layer_idx}.head_{h}"
            if key not in tensors:
                continue
            vec = _resize_rel_pos_vector(tensors[key], p.numel()).to(device=p.device, dtype=p.dtype)
            p.data.copy_(vec)
            loaded += 1
    return loaded
