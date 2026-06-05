"""Load trained per-head relative-position bias into Qwen3-VL for inference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_train_sparse():
    root = _repo_root()
    train_dir = str(root / "train")
    for p in (str(root), train_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from patch_sparse_attn import (
        _iter_text_attention_modules,
        load_sparse_rel_pos_checkpoint,
        patch_model_for_sparse_eval,
        unpack_sparse_rel_pos_checkpoint,
    )

    return (
        _iter_text_attention_modules,
        patch_model_for_sparse_eval,
        load_sparse_rel_pos_checkpoint,
        unpack_sparse_rel_pos_checkpoint,
    )


def setup_sparse_attention_for_eval(
    model: torch.nn.Module,
    *,
    rel_pos_path: str,
    rel_pos_buckets: int = 16384,
    layer_id: int = 0,
    save_attn_scores_dir: str | None = None,
) -> None:
    """Eval: one layer uses sparse pre-RoPE attention; others keep default."""
    (
        _iter_text_attention_modules,
        patch_model_for_sparse_eval,
        load_sparse_rel_pos_checkpoint,
        unpack_sparse_rel_pos_checkpoint,
    ) = _import_train_sparse()
    patch_model_for_sparse_eval(model, layer_id=layer_id, num_buckets=rel_pos_buckets)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for attn in _iter_text_attention_modules(model):
        if hasattr(attn, "rel_pos_bias_per_head"):
            attn.rel_pos_bias_per_head.to(device=device, dtype=dtype)
    loaded = load_sparse_rel_pos_checkpoint(
        model,
        rel_pos_path,
        expected_layer_id=layer_id,
    )
    if loaded == 0:
        raise RuntimeError(f"No rel-pos vectors loaded from {rel_pos_path}")
    if save_attn_scores_dir:
        from attn_score_dump import enable_attn_score_dump

        enable_attn_score_dump(model, save_attn_scores_dir)
    from loguru import logger

    logger.info(
        f"[sparse_attn] eval layer_id={layer_id}: f(d)*Q_pre*K_pre/sqrt(d) prefill+decode | "
        f"loaded {loaded} head vectors from {rel_pos_path}",
    )
