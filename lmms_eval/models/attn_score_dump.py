"""Enable pre-softmax attention score dump for Qwen3-VL eval."""

from __future__ import annotations

import sys
from pathlib import Path


def _import_train_attn_dump():
    root = Path(__file__).resolve().parents[2]
    train_dir = str(root / "train")
    for p in (str(root), train_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from attn_score_dump import (
        begin_attn_score_sample,
        enable_attn_score_dump,
        flush_attn_score_sample,
        patch_baseline_attn_dump,
    )

    return begin_attn_score_sample, enable_attn_score_dump, flush_attn_score_sample, patch_baseline_attn_dump


def setup_attn_score_dump(model, dump_dir: str, *, sparse_patched: bool) -> None:
    begin, enable, flush, patch_baseline = _import_train_attn_dump()
    enable(model, dump_dir)
    if not sparse_patched:
        patch_baseline(model)


def begin_sample(model, sample_idx: int) -> None:
    begin, _, _, _ = _import_train_attn_dump()
    begin(model, sample_idx)


def flush_sample(model) -> None:
    _, _, flush, _ = _import_train_attn_dump()
    flush(model)
