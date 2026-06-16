#!/usr/bin/env python3
"""
Pre-RoPE key coverage in post-RoPE softmax attention.

Terminology:
- rel_distance d = q_pos - k_pos (excel column "distance")
- key_index k = key position in sequence (0 .. kv_len-1); for fixed q, k = q - d

Two key-index sets (union when distance excel is provided):
1. pre-RoPE Q·K softmax top-k: directly pick ceil(ratio*n_valid) key indices
2. distance excel: pick ceil(ratio*n_valid) rel_distance values by count, map to k=q-d

Stages (--stage): last | all | decode (same semantics as distance calibration).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = Path(__file__).resolve().parents[1]
CALIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(CALIB_DIR))

from collator import Qwen3VLDataCollator, _vision_encode
from dataset import (
    DEFAULT_MAX_PIXELS,
    TomatoSFTDataset,
    load_tomato_split,
    resolve_min_pixels,
)
from patch_sparse_attn import _patch_causal_lm_sparse_forward, _patch_text_model_early_stop
from sparse_attention import qk_pre_softmax_scores

from top_p_distance_calibrate import (
    _build_valid_mask,
    _get_language_model,
    _iter_text_attention_modules,
    _q_positions_from_pad_mask,
)


def _set_post_mass_pad_mask(model: torch.nn.Module, pad_mask_2d: torch.Tensor | None) -> None:
    for attn in _iter_text_attention_modules(model):
        if getattr(attn, "_calib_post_mass_patched", False) or getattr(
            attn, "_calib_post_mass_decode_patched", False
        ):
            attn._calib_pad_mask_2d = pad_mask_2d


def _query_row_mask_bq(
    batch_size: int,
    q_len: int,
    q_positions: torch.Tensor,
    query_scope: str,
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device, dtype=torch.long).view(1, q_len)
    last_q = q_positions.to(device=device).view(batch_size, 1)
    if query_scope == "last":
        return q_idx == last_q
    if query_scope == "all":
        return q_idx <= last_q
    raise ValueError(f"unknown query_scope={query_scope!r}")


def _softmax_probs_bhqk(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~valid, neg_inf)
    return F.softmax(masked.float(), dim=-1).to(scores.dtype)


def load_head_ranked_rel_distances_from_excel(
    excel_dir: Path,
    *,
    n_heads: int,
) -> dict[int, list[int]]:
    """Per head: rel_distance d=q_pos-k_pos values, sorted by excel count desc."""
    import pandas as pd

    out: dict[int, list[int]] = {}
    for h in range(n_heads):
        path = excel_dir / f"head_{h:02d}.xlsx"
        if not path.exists():
            continue
        df = pd.read_excel(path)
        if df.empty:
            continue
        df = df.sort_values("count", ascending=False)
        out[h] = [int(d) for d in df["distance"].tolist()]
    return out


def _build_rel_dist_prefill(
    batch_size: int,
    n_heads: int,
    q_len: int,
    kv_len: int,
    q_positions: torch.Tensor,
    query_scope: str,
    device: torch.device,
) -> torch.Tensor:
    if query_scope == "last":
        last_q = q_positions.to(device=device)
        k_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
        rel = last_q.view(batch_size, 1, 1, 1) - k_idx
        return rel.expand(batch_size, n_heads, 1, kv_len)
    q_idx = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
    k_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
    return (q_idx - k_idx).expand(batch_size, n_heads, q_len, kv_len)


def _build_rel_dist_decode(
    batch_size: int,
    n_heads: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    k_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
    q_abs = kv_len - 1
    rel = q_abs - k_idx
    return rel.expand(batch_size, n_heads, 1, kv_len)


def _key_index_mask_from_rel_distance_topk_bhqk(
    rel_dist_bhqk: torch.Tensor,
    valid_bhqk: torch.Tensor,
    head_ranked_rel_distances: dict[int, list[int]],
    rel_distance_topk_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map top rel_distance values (from excel) to key-index mask; ratio=0 selects none."""
    n_heads = rel_dist_bhqk.size(1)
    valid_h = _expand_valid_bhqk(valid_bhqk, n_heads)
    n_valid = valid_h.sum(dim=-1)
    if rel_distance_topk_ratio <= 0:
        return torch.zeros_like(valid_h), torch.zeros_like(n_valid, dtype=torch.float32)
    key_index_mask = torch.zeros_like(valid_h)
    n_rel_dist_kept = torch.ceil(n_valid.float() * float(rel_distance_topk_ratio)).long().clamp(min=1)
    for h, ranked_rel_d in head_ranked_rel_distances.items():
        if not ranked_rel_d:
            continue
        rel_h = rel_dist_bhqk[:, h].long()
        max_r = min(int(n_rel_dist_kept[:, h, :].max().item()), len(ranked_rel_d))
        match_h = torch.zeros_like(valid_h[:, h], dtype=torch.bool)
        for r in range(max_r):
            d = ranked_rel_d[r]
            in_top_r = n_rel_dist_kept[:, h, :] > r
            match_h |= (rel_h == d) & in_top_r.unsqueeze(-1)
        key_index_mask[:, h] = match_h
    return key_index_mask & valid_h, n_rel_dist_kept.float()


def _expand_valid_bhqk(valid_bhqk: torch.Tensor, n_heads: int) -> torch.Tensor:
    if valid_bhqk.size(1) == 1:
        return valid_bhqk.expand(-1, n_heads, -1, -1)
    return valid_bhqk


def _topk_mask_pre_softmax_bhqk(
    score_pre: torch.Tensor,
    valid_bhqk: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Per (B,H,Q) row: keep top ceil(ratio * n_valid) key indices by pre-RoPE softmax (0=none)."""
    n_heads = score_pre.size(1)
    valid_h = _expand_valid_bhqk(valid_bhqk, n_heads)
    if ratio <= 0:
        return torch.zeros_like(valid_h)
    probs = _softmax_probs_bhqk(score_pre, valid_h)
    neg_inf = torch.finfo(probs.dtype).min
    masked = probs.masked_fill(~valid_h, neg_inf)
    n_valid = valid_h.sum(dim=-1, keepdim=True).clamp(min=1)
    k_keep = torch.ceil(n_valid.float() * float(ratio)).long().clamp(min=1)
    max_k = int(k_keep.max().item())
    _, idx = torch.topk(masked, max_k, dim=-1)
    ranks = torch.arange(max_k, device=valid_h.device).view(1, 1, 1, -1)
    in_topk = ranks < k_keep
    topk = torch.zeros_like(valid_h)
    topk.scatter_(dim=-1, index=idx, src=in_topk)
    return topk & valid_h


def _post_mass_core(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    valid_bhqk: torch.Tensor,
    *,
    topk_ratio: float,
    rel_dist_bhqk: torch.Tensor | None = None,
    head_ranked_rel_distances: dict[int, list[int]] | None = None,
    rel_distance_topk_ratio: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    n_heads = score_pre.size(1)
    valid_h = _expand_valid_bhqk(valid_bhqk, n_heads)
    qk_key_index_mask = _topk_mask_pre_softmax_bhqk(score_pre, valid_h, topk_ratio)
    post_probs = _softmax_probs_bhqk(score_post, valid_h)
    mass_qk = (post_probs * qk_key_index_mask.float()).sum(dim=-1)
    cnt_qk_key_index = qk_key_index_mask.sum(dim=-1).float()

    use_dist = bool(
        head_ranked_rel_distances
        and rel_dist_bhqk is not None
        and rel_distance_topk_ratio is not None
        and rel_distance_topk_ratio > 0
    )
    use_qk = topk_ratio > 0

    if not use_dist:
        if use_qk:
            union_key_index_mask = qk_key_index_mask & valid_h
            mass_union = (post_probs * union_key_index_mask.float()).sum(dim=-1)
            cnt_union_key_index = union_key_index_mask.sum(dim=-1).float()
            return mass_qk, cnt_qk_key_index, None, None, None, mass_union, cnt_union_key_index
        return mass_qk, cnt_qk_key_index, None, None, None, None, None

    dist_key_index_mask, cnt_rel_dist_topk = _key_index_mask_from_rel_distance_topk_bhqk(
        rel_dist_bhqk, valid_h, head_ranked_rel_distances, rel_distance_topk_ratio
    )
    mass_dist = (post_probs * dist_key_index_mask.float()).sum(dim=-1)
    cnt_dist_key_index = dist_key_index_mask.sum(dim=-1).float()
    union_key_index_mask = torch.zeros_like(valid_h)
    if use_qk:
        union_key_index_mask |= qk_key_index_mask
    union_key_index_mask |= dist_key_index_mask
    union_key_index_mask &= valid_h
    mass_union = (post_probs * union_key_index_mask.float()).sum(dim=-1)
    cnt_union_key_index = union_key_index_mask.sum(dim=-1).float()
    return (
        mass_qk,
        cnt_qk_key_index,
        mass_dist,
        cnt_dist_key_index,
        cnt_rel_dist_topk,
        mass_union,
        cnt_union_key_index,
    )


def _head_row_aggregate(
    values_bhq: torch.Tensor,
    row_mask_bq: torch.Tensor,
) -> tuple[dict[int, float], dict[int, int]]:
    """values (B,H,Q); row_mask (B,Q) -> per-head sum and row count."""
    n_heads = values_bhq.size(1)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for h in range(n_heads):
        sel = values_bhq[:, h, :][row_mask_bq]
        if sel.numel() == 0:
            continue
        sums[h] = float(sel.sum().item())
        counts[h] = int(sel.numel())
    return sums, counts


def _head_coverage_sums(
    coverage_bhq: torch.Tensor,
    row_mask_bq: torch.Tensor,
) -> tuple[dict[int, float], dict[int, int]]:
    """coverage (B,H,Q); row_mask (B,Q) -> per-head sum and count."""
    return _head_row_aggregate(coverage_bhq, row_mask_bq)


def compute_prefill_post_mass(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    q_positions: torch.Tensor,
    *,
    topk_ratio: float,
    query_scope: str,
    pad_mask_2d: torch.Tensor | None = None,
    head_ranked_rel_distances: dict[int, list[int]] | None = None,
    rel_distance_topk_ratio: float | None = None,
) -> tuple[
    dict[int, float],
    dict[int, int],
    dict[int, float],
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
]:
    batch_size, n_heads, q_len, kv_len = score_pre.shape
    device = score_pre.device

    if query_scope == "last":
        batch_idx = torch.arange(batch_size, device=device)
        last_q = q_positions.to(device=device)
        rows_pre = score_pre[batch_idx, :, last_q, :]
        rows_post = score_post[batch_idx, :, last_q, :]
        k_idx = torch.arange(kv_len, device=device).view(1, 1, kv_len)
        valid = k_idx <= last_q.view(batch_size, 1, 1)
        if pad_mask_2d is not None:
            valid = valid & pad_mask_2d.to(device=device).bool().view(batch_size, 1, kv_len)
        valid_bhqk = valid.unsqueeze(1)
        rel_dist = _build_rel_dist_prefill(
            batch_size, n_heads, 1, kv_len, q_positions, "last", device
        )
        (
            mass_qk,
            cnt_qk_key_index,
            mass_dist,
            cnt_dist_key_index,
            cnt_rel_dist_topk,
            mass_union,
            cnt_union_key_index,
        ) = _post_mass_core(
            rows_pre.unsqueeze(2),
            rows_post.unsqueeze(2),
            valid_bhqk,
            topk_ratio=topk_ratio,
            rel_dist_bhqk=rel_dist,
            head_ranked_rel_distances=head_ranked_rel_distances,
            rel_distance_topk_ratio=rel_distance_topk_ratio,
        )
        mass_qk = mass_qk.squeeze(2)
        cnt_qk_key_index = cnt_qk_key_index.squeeze(2)
        mass_sums = {h: float(mass_qk[:, h].sum().item()) for h in range(n_heads)}
        mass_counts = {h: batch_size for h in range(n_heads)}
        key_sums = {h: float(cnt_qk_key_index[:, h].sum().item()) for h in range(n_heads)}
        dist_sums, dist_key_sums, rel_dist_topk_sums, union_sums, union_key_sums = (
            _pack_optional_mass(
                mass_dist,
                cnt_dist_key_index,
                cnt_rel_dist_topk,
                mass_union,
                cnt_union_key_index,
                n_heads,
                batch_size,
            )
        )
        return (
            mass_sums,
            mass_counts,
            key_sums,
            dist_sums,
            dist_key_sums,
            rel_dist_topk_sums,
            union_sums,
            union_key_sums,
        )

    valid = _build_valid_mask(
        batch_size, q_len, kv_len, q_positions, query_scope, device, pad_mask_2d=pad_mask_2d
    )
    rel_dist = _build_rel_dist_prefill(
        batch_size, n_heads, q_len, kv_len, q_positions, query_scope, device
    )
    (
        mass_qk,
        cnt_qk_key_index,
        mass_dist,
        cnt_dist_key_index,
        cnt_rel_dist_topk,
        mass_union,
        cnt_union_key_index,
    ) = _post_mass_core(
        score_pre,
        score_post,
        valid,
        topk_ratio=topk_ratio,
        rel_dist_bhqk=rel_dist,
        head_ranked_rel_distances=head_ranked_rel_distances,
        rel_distance_topk_ratio=rel_distance_topk_ratio,
    )
    row_mask = _query_row_mask_bq(batch_size, q_len, q_positions, query_scope, device)
    mass_sums, mass_counts = _head_row_aggregate(mass_qk, row_mask)
    key_sums, _ = _head_row_aggregate(cnt_qk_key_index, row_mask)
    dist_sums = dist_key_sums = rel_dist_topk_sums = union_sums = union_key_sums = None
    if mass_dist is not None:
        dist_sums, _ = _head_row_aggregate(mass_dist, row_mask)
        dist_key_sums, _ = _head_row_aggregate(cnt_dist_key_index, row_mask)
        rel_dist_topk_sums, _ = _head_row_aggregate(cnt_rel_dist_topk, row_mask)
        union_sums, _ = _head_row_aggregate(mass_union, row_mask)
        union_key_sums, _ = _head_row_aggregate(cnt_union_key_index, row_mask)
    return (
        mass_sums,
        mass_counts,
        key_sums,
        dist_sums,
        dist_key_sums,
        rel_dist_topk_sums,
        union_sums,
        union_key_sums,
    )


def _pack_optional_mass(
    mass_dist: torch.Tensor | None,
    cnt_dist_key_index: torch.Tensor | None,
    cnt_rel_dist_topk: torch.Tensor | None,
    mass_union: torch.Tensor | None,
    cnt_union_key_index: torch.Tensor | None,
    n_heads: int,
    batch_size: int,
) -> tuple[
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
]:
    if mass_dist is None:
        return None, None, None, None, None
    mass_dist = mass_dist.squeeze(2)
    cnt_dist_key_index = cnt_dist_key_index.squeeze(2)
    cnt_rel_dist_topk = cnt_rel_dist_topk.squeeze(2)
    mass_union = mass_union.squeeze(2)
    cnt_union_key_index = cnt_union_key_index.squeeze(2)
    dist_sums = {h: float(mass_dist[:, h].sum().item()) for h in range(n_heads)}
    dist_key_sums = {h: float(cnt_dist_key_index[:, h].sum().item()) for h in range(n_heads)}
    rel_dist_topk_sums = {h: float(cnt_rel_dist_topk[:, h].sum().item()) for h in range(n_heads)}
    union_sums = {h: float(mass_union[:, h].sum().item()) for h in range(n_heads)}
    union_key_sums = {h: float(cnt_union_key_index[:, h].sum().item()) for h in range(n_heads)}
    return dist_sums, dist_key_sums, rel_dist_topk_sums, union_sums, union_key_sums


def _build_decode_valid_mask(
    batch_size: int,
    kv_len: int,
    q_abs: torch.Tensor,
    pad_mask_2d: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    k_idx = torch.arange(kv_len, device=device).view(1, 1, kv_len)
    valid = k_idx <= q_abs.to(device=device).view(batch_size, 1, 1)
    if pad_mask_2d is not None:
        pm = pad_mask_2d.to(device=device).bool()
        if pm.size(-1) < kv_len:
            pm = F.pad(pm, (0, kv_len - pm.size(-1)), value=False)
        else:
            pm = pm[:, :kv_len]
        batch_idx = torch.arange(batch_size, device=device)
        q_col = q_abs.to(device=device).clamp(max=kv_len - 1)
        pm[batch_idx, q_col] = True
        valid = valid & pm.view(batch_size, 1, kv_len)
    return valid.unsqueeze(1)


def compute_decode_post_mass(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    *,
    topk_ratio: float,
    pad_mask_2d: torch.Tensor | None,
    head_ranked_rel_distances: dict[int, list[int]] | None = None,
    rel_distance_topk_ratio: float | None = None,
) -> tuple[
    dict[int, float],
    dict[int, int],
    dict[int, float],
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
]:
    if score_pre.size(-2) != 1:
        raise ValueError(f"expected q_len==1 for decode, got {score_pre.size(-2)}")

    batch_size = score_pre.size(0)
    n_heads = score_pre.size(1)
    kv_len = score_pre.size(-1)
    device = score_pre.device
    q_abs = torch.full((batch_size,), kv_len - 1, dtype=torch.long, device=device)

    rows_pre = score_pre[:, :, 0, :]
    rows_post = score_post[:, :, 0, :]
    valid_bhqk = _build_decode_valid_mask(batch_size, kv_len, q_abs, pad_mask_2d, device)
    rel_dist = _build_rel_dist_decode(batch_size, n_heads, kv_len, device)
    (
        mass_qk,
        cnt_qk_key_index,
        mass_dist,
        cnt_dist_key_index,
        cnt_rel_dist_topk,
        mass_union,
        cnt_union_key_index,
    ) = _post_mass_core(
        rows_pre.unsqueeze(2),
        rows_post.unsqueeze(2),
        valid_bhqk,
        topk_ratio=topk_ratio,
        rel_dist_bhqk=rel_dist,
        head_ranked_rel_distances=head_ranked_rel_distances,
        rel_distance_topk_ratio=rel_distance_topk_ratio,
    )
    mass_qk = mass_qk.squeeze(2)
    cnt_qk_key_index = cnt_qk_key_index.squeeze(2)
    mass_sums = {h: float(mass_qk[:, h].sum().item()) for h in range(n_heads)}
    mass_counts = {h: batch_size for h in range(n_heads)}
    key_sums = {h: float(cnt_qk_key_index[:, h].sum().item()) for h in range(n_heads)}
    dist_sums, dist_key_sums, rel_dist_topk_sums, union_sums, union_key_sums = _pack_optional_mass(
        mass_dist,
        cnt_dist_key_index,
        cnt_rel_dist_topk,
        mass_union,
        cnt_union_key_index,
        n_heads,
        batch_size,
    )
    return (
        mass_sums,
        mass_counts,
        key_sums,
        dist_sums,
        dist_key_sums,
        rel_dist_topk_sums,
        union_sums,
        union_key_sums,
    )


def stats_per_head_from_capture(
    capture: dict, layer_id: int
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
    dict[int, float] | None,
]:
    data = capture.get(layer_id, {})
    head_sums = data.get("head_sums", {})
    head_counts = data.get("head_counts", {})
    head_key_sums = data.get("head_key_sums", {})
    head_dist_sums = data.get("head_dist_sums")
    head_dist_key_index_sums = data.get("head_dist_key_index_sums")
    head_rel_dist_topk_sums = data.get("head_rel_dist_topk_sums")
    head_union_sums = data.get("head_union_sums")
    head_union_key_index_sums = data.get("head_union_key_index_sums")
    mass: dict[int, float] = {}
    key_n: dict[int, float] = {}
    dist_mass: dict[int, float] | None = {} if head_dist_sums is not None else None
    rel_dist_n: dict[int, float] | None = {} if head_rel_dist_topk_sums is not None else None
    dist_key_n: dict[int, float] | None = {} if head_dist_key_index_sums is not None else None
    union_mass: dict[int, float] | None = {} if head_union_sums is not None else None
    union_n: dict[int, float] | None = {} if head_union_key_index_sums is not None else None
    for h, s in head_sums.items():
        c = int(head_counts.get(h, 0))
        if c > 0:
            mass[int(h)] = float(s) / c
            key_n[int(h)] = float(head_key_sums.get(h, 0.0)) / c
            if dist_mass is not None and head_dist_sums is not None:
                dist_mass[int(h)] = float(head_dist_sums.get(h, 0.0)) / c
            if rel_dist_n is not None and head_rel_dist_topk_sums is not None:
                rel_dist_n[int(h)] = float(head_rel_dist_topk_sums.get(h, 0.0)) / c
            if dist_key_n is not None and head_dist_key_index_sums is not None:
                dist_key_n[int(h)] = float(head_dist_key_index_sums.get(h, 0.0)) / c
            if union_mass is not None and head_union_sums is not None:
                union_mass[int(h)] = float(head_union_sums.get(h, 0.0)) / c
            if union_n is not None and head_union_key_index_sums is not None:
                union_n[int(h)] = float(head_union_key_index_sums.get(h, 0.0)) / c
    return mass, key_n, dist_mass, rel_dist_n, dist_key_n, union_mass, union_n


def _layer_mean_values(
    *,
    qk_ratios: dict[int, float],
    n_heads: int,
    qk_key_index_counts: dict[int, float] | None = None,
    dist_ratios: dict[int, float] | None = None,
    rel_dist_topk_counts: dict[int, float] | None = None,
    matched_key_index_counts: dict[int, float] | None = None,
    union_ratios: dict[int, float] | None = None,
    union_key_index_counts: dict[int, float] | None = None,
) -> dict[str, float]:
    qk_vals = [qk_ratios[h] for h in range(n_heads) if h in qk_ratios]
    mean_qk = sum(qk_vals) / len(qk_vals) if qk_vals else float("nan")

    mean_qk_n = float("nan")
    if qk_key_index_counts is not None:
        qk_n_vals = [qk_key_index_counts[h] for h in range(n_heads) if h in qk_key_index_counts]
        mean_qk_n = sum(qk_n_vals) / len(qk_n_vals) if qk_n_vals else float("nan")

    mean_dist = mean_rel_dist_n = mean_matched_key_n = mean_union = mean_union_n = float("nan")
    if dist_ratios is not None:
        dist_vals = [dist_ratios[h] for h in range(n_heads) if h in dist_ratios]
        mean_dist = sum(dist_vals) / len(dist_vals) if dist_vals else float("nan")
    if rel_dist_topk_counts is not None:
        rel_vals = [rel_dist_topk_counts[h] for h in range(n_heads) if h in rel_dist_topk_counts]
        mean_rel_dist_n = sum(rel_vals) / len(rel_vals) if rel_vals else float("nan")
    if matched_key_index_counts is not None:
        key_vals = [
            matched_key_index_counts[h] for h in range(n_heads) if h in matched_key_index_counts
        ]
        mean_matched_key_n = sum(key_vals) / len(key_vals) if key_vals else float("nan")
    if union_ratios is not None:
        union_vals = [union_ratios[h] for h in range(n_heads) if h in union_ratios]
        mean_union = sum(union_vals) / len(union_vals) if union_vals else float("nan")
    if union_key_index_counts is not None:
        union_n_vals = [
            union_key_index_counts[h] for h in range(n_heads) if h in union_key_index_counts
        ]
        mean_union_n = sum(union_n_vals) / len(union_n_vals) if union_n_vals else float("nan")

    out = {
        "post_mass_on_pre_qk_topk": mean_qk,
        "pre_qk_topk_key_index_count": mean_qk_n,
    }
    if dist_ratios is not None:
        out["post_mass_on_rel_distance_excel"] = mean_dist
        out["rel_distance_topk_count"] = mean_rel_dist_n
        out["matched_key_index_count"] = mean_matched_key_n
        out["post_mass_on_union"] = mean_union
        out["union_key_index_count"] = mean_union_n
    return out


def _format_layer_mean_line(values: dict[str, float], *, has_dist: bool) -> str:
    line = (
        f"post_mass_on_pre_qk_topk={values['post_mass_on_pre_qk_topk']:.6f} "
        f"pre_qk_topk_key_index_count={values['pre_qk_topk_key_index_count']:.1f}"
    )
    if has_dist:
        line += (
            f" post_mass_on_rel_distance_excel={values['post_mass_on_rel_distance_excel']:.6f}"
            f" rel_distance_topk_count={values['rel_distance_topk_count']:.1f}"
            f" matched_key_index_count={values['matched_key_index_count']:.1f}"
            f" post_mass_on_union={values['post_mass_on_union']:.6f}"
            f" union_key_index_count={values['union_key_index_count']:.1f}"
        )
    return line


def _layer_mean_line(
    *,
    qk_ratios: dict[int, float],
    n_heads: int,
    qk_key_index_counts: dict[int, float] | None = None,
    dist_ratios: dict[int, float] | None = None,
    rel_dist_topk_counts: dict[int, float] | None = None,
    matched_key_index_counts: dict[int, float] | None = None,
    union_ratios: dict[int, float] | None = None,
    union_key_index_counts: dict[int, float] | None = None,
) -> str:
    values = _layer_mean_values(
        qk_ratios=qk_ratios,
        n_heads=n_heads,
        qk_key_index_counts=qk_key_index_counts,
        dist_ratios=dist_ratios,
        rel_dist_topk_counts=rel_dist_topk_counts,
        matched_key_index_counts=matched_key_index_counts,
        union_ratios=union_ratios,
        union_key_index_counts=union_key_index_counts,
    )
    return _format_layer_mean_line(values, has_dist=dist_ratios is not None)


class AllSamplesLayerMeanAccumulator:
    """Running mean of per-sample layer_mean (each sample/batch counts equally)."""

    def __init__(self, *, has_dist: bool) -> None:
        self.has_dist = has_dist
        self.n = 0
        self._sums: dict[str, float] = {
            "post_mass_on_pre_qk_topk": 0.0,
            "pre_qk_topk_key_index_count": 0.0,
        }
        if has_dist:
            self._sums.update(
                {
                    "post_mass_on_rel_distance_excel": 0.0,
                    "rel_distance_topk_count": 0.0,
                    "matched_key_index_count": 0.0,
                    "post_mass_on_union": 0.0,
                    "union_key_index_count": 0.0,
                }
            )

    def add(
        self,
        *,
        qk_ratios: dict[int, float],
        n_heads: int,
        qk_key_index_counts: dict[int, float] | None = None,
        dist_ratios: dict[int, float] | None = None,
        rel_dist_topk_counts: dict[int, float] | None = None,
        matched_key_index_counts: dict[int, float] | None = None,
        union_ratios: dict[int, float] | None = None,
        union_key_index_counts: dict[int, float] | None = None,
    ) -> None:
        values = _layer_mean_values(
            qk_ratios=qk_ratios,
            n_heads=n_heads,
            qk_key_index_counts=qk_key_index_counts,
            dist_ratios=dist_ratios,
            rel_dist_topk_counts=rel_dist_topk_counts,
            matched_key_index_counts=matched_key_index_counts,
            union_ratios=union_ratios,
            union_key_index_counts=union_key_index_counts,
        )
        for k in self._sums:
            v = values.get(k, float("nan"))
            if not math.isnan(v):
                self._sums[k] += v
        self.n += 1

    def mean(self) -> dict[str, float]:
        if self.n == 0:
            return {k: float("nan") for k in self._sums}
        return {k: s / self.n for k, s in self._sums.items()}


def print_sample_head_coverage(
    *,
    sample_label: str,
    layer_id: int,
    n_heads: int,
    qk_ratios: dict[int, float],
    print_mode: str = "full",
    qk_key_index_counts: dict[int, float] | None = None,
    dist_ratios: dict[int, float] | None = None,
    rel_dist_topk_counts: dict[int, float] | None = None,
    matched_key_index_counts: dict[int, float] | None = None,
    union_ratios: dict[int, float] | None = None,
    union_key_index_counts: dict[int, float] | None = None,
) -> None:
    if print_mode == "full":
        qk_parts = [f"h{h}={qk_ratios.get(h, float('nan')):.6f}" for h in range(n_heads)]
        print(
            f"[post-mass] {sample_label} layer={layer_id} post_mass_on_pre_qk_topk: "
            + " ".join(qk_parts),
            flush=True,
        )
        if qk_key_index_counts is not None:
            qk_n_parts = [
                f"h{h}={qk_key_index_counts.get(h, float('nan')):.1f}" for h in range(n_heads)
            ]
            print(
                f"[post-mass] {sample_label} layer={layer_id} pre_qk_topk_key_index_count: "
                + " ".join(qk_n_parts),
                flush=True,
            )
        if dist_ratios is not None:
            dist_parts = [f"h{h}={dist_ratios.get(h, float('nan')):.6f}" for h in range(n_heads)]
            print(
                f"[post-mass] {sample_label} layer={layer_id} post_mass_on_rel_distance_excel: "
                + " ".join(dist_parts),
                flush=True,
            )
        if rel_dist_topk_counts is not None:
            rel_parts = [
                f"h{h}={rel_dist_topk_counts.get(h, float('nan')):.1f}" for h in range(n_heads)
            ]
            print(
                f"[post-mass] {sample_label} layer={layer_id} rel_distance_topk_count: "
                + " ".join(rel_parts),
                flush=True,
            )
        if matched_key_index_counts is not None:
            key_parts = [
                f"h{h}={matched_key_index_counts.get(h, float('nan')):.1f}" for h in range(n_heads)
            ]
            print(
                f"[post-mass] {sample_label} layer={layer_id} matched_key_index_count: "
                + " ".join(key_parts),
                flush=True,
            )
        if union_ratios is not None:
            union_parts = [f"h{h}={union_ratios.get(h, float('nan')):.6f}" for h in range(n_heads)]
            print(
                f"[post-mass] {sample_label} layer={layer_id} post_mass_on_union: "
                + " ".join(union_parts),
                flush=True,
            )
        if union_key_index_counts is not None:
            union_n_parts = [
                f"h{h}={union_key_index_counts.get(h, float('nan')):.1f}" for h in range(n_heads)
            ]
            print(
                f"[post-mass] {sample_label} layer={layer_id} union_key_index_count: "
                + " ".join(union_n_parts),
                flush=True,
            )

    layer_mean = _layer_mean_line(
        qk_ratios=qk_ratios,
        n_heads=n_heads,
        qk_key_index_counts=qk_key_index_counts,
        dist_ratios=dist_ratios,
        rel_dist_topk_counts=rel_dist_topk_counts,
        matched_key_index_counts=matched_key_index_counts,
        union_ratios=union_ratios,
        union_key_index_counts=union_key_index_counts,
    )
    print(f"[post-mass] {sample_label} layer={layer_id} layer_mean: {layer_mean}", flush=True)
    print("-" * 80, flush=True)


def print_all_samples_summary(
    sample_layer_mean_acc: AllSamplesLayerMeanAccumulator,
    *,
    layer_id: int,
) -> None:
    if sample_layer_mean_acc.n <= 0:
        return
    layer_mean = _format_layer_mean_line(
        sample_layer_mean_acc.mean(),
        has_dist=sample_layer_mean_acc.has_dist,
    )
    print(
        f"[post-mass] all_samples (n={sample_layer_mean_acc.n}) "
        f"layer={layer_id} layer_mean: {layer_mean}",
        flush=True,
    )
    print("=" * 80, flush=True)


def install_prefill_post_mass_hooks(
    model: torch.nn.Module,
    capture: dict,
    *,
    layer_id: int,
    topk_ratio: float,
    query_scope: str,
    head_ranked_rel_distances: dict[int, list[int]] | None = None,
    rel_distance_topk_ratio: float | None = None,
) -> None:
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        if getattr(attn, "_calib_post_mass_patched", False):
            continue

        orig_forward = attn.forward

        def make_forward(
            _layer_idx: int,
            _orig_forward,
            _topk_ratio: float,
            _query_scope: str,
            _head_ranked_rel_distances,
            _rel_distance_topk_ratio,
        ):
            def forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                query_pre = query_states
                key_pre = key_states

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                q_len = query_states.size(-2)
                kv_len = key_states.size(-2)
                if q_len == kv_len and q_len > 0:
                    prev_kv = int(capture.get(_layer_idx, {}).get("kv_len", -1))
                    if kv_len >= prev_kv:
                        score_pre = qk_pre_softmax_scores(
                            self, query_pre, key_pre, attention_mask, self.scaling
                        )
                        score_post = qk_pre_softmax_scores(
                            self, query_states, key_states, attention_mask, self.scaling
                        )
                        batch_size = score_pre.size(0)
                        pad_mask_2d = getattr(self, "_calib_pad_mask_2d", None)
                        q_positions = _q_positions_from_pad_mask(
                            pad_mask_2d,
                            batch_size=batch_size,
                            kv_len=kv_len,
                            device=score_pre.device,
                        )
                        (
                            head_sums,
                            head_counts,
                            head_key_sums,
                            head_dist_sums,
                            head_dist_key_index_sums,
                            head_rel_dist_topk_sums,
                            head_union_sums,
                            head_union_key_index_sums,
                        ) = compute_prefill_post_mass(
                            score_pre,
                            score_post,
                            q_positions,
                            topk_ratio=_topk_ratio,
                            query_scope=_query_scope,
                            pad_mask_2d=pad_mask_2d,
                            head_ranked_rel_distances=_head_ranked_rel_distances,
                            rel_distance_topk_ratio=_rel_distance_topk_ratio,
                        )
                        del score_pre, score_post
                        capture[_layer_idx] = {
                            "head_sums": head_sums,
                            "head_counts": head_counts,
                            "head_key_sums": head_key_sums,
                            "head_dist_sums": head_dist_sums,
                            "head_dist_key_index_sums": head_dist_key_index_sums,
                            "head_rel_dist_topk_sums": head_rel_dist_topk_sums,
                            "head_union_sums": head_union_sums,
                            "head_union_key_index_sums": head_union_key_index_sums,
                            "batch_size": batch_size,
                            "kv_len": kv_len,
                        }

                return _orig_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )

            return forward

        attn.forward = make_forward(
            layer_idx, orig_forward, topk_ratio, query_scope, head_ranked_rel_distances, rel_distance_topk_ratio
        ).__get__(attn, type(attn))
        attn._calib_post_mass_patched = True


def install_decode_post_mass_hooks(
    model: torch.nn.Module,
    capture: dict,
    *,
    layer_id: int,
    topk_ratio: float,
    head_ranked_rel_distances: dict[int, list[int]] | None = None,
    rel_distance_topk_ratio: float | None = None,
) -> None:
    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        if getattr(attn, "_calib_post_mass_decode_patched", False):
            continue

        orig_forward = attn.forward

        def make_forward(
            _layer_idx: int,
            _orig_forward,
            _topk_ratio: float,
            _model: torch.nn.Module,
            _head_ranked_rel_distances,
            _rel_distance_topk_ratio,
        ):
            def forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                query_pre = query_states
                key_pre_step = key_states

                cos, sin = position_embeddings
                query_states, key_post_step = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                step_len = key_pre_step.size(-2)
                if step_len > 1:
                    self._calib_post_mass_pre_key_cache = key_pre_step.detach()
                    self._calib_post_mass_post_key_cache = key_post_step.detach()
                    key_pre = key_pre_step
                    key_post = key_post_step
                else:
                    cached_pre = getattr(self, "_calib_post_mass_pre_key_cache", None)
                    cached_post = getattr(self, "_calib_post_mass_post_key_cache", None)
                    key_pre = (
                        torch.cat([cached_pre, key_pre_step], dim=-2)
                        if cached_pre is not None
                        else key_pre_step
                    )
                    key_post = (
                        torch.cat([cached_post, key_post_step], dim=-2)
                        if cached_post is not None
                        else key_post_step
                    )

                q_len = query_states.size(-2)
                kv_len = key_post.size(-2)
                if (
                    q_len == 1
                    and kv_len > 1
                    and not getattr(_model, "_calib_post_mass_captured", False)
                ):
                    score_pre = qk_pre_softmax_scores(
                        self, query_pre, key_pre, attention_mask, self.scaling
                    )
                    score_post = qk_pre_softmax_scores(
                        self, query_states, key_post, attention_mask, self.scaling
                    )
                    pad_mask_2d = getattr(self, "_calib_pad_mask_2d", None)
                    (
                        head_sums,
                        head_counts,
                        head_key_sums,
                        head_dist_sums,
                        head_dist_key_index_sums,
                        head_rel_dist_topk_sums,
                        head_union_sums,
                        head_union_key_index_sums,
                    ) = compute_decode_post_mass(
                        score_pre,
                        score_post,
                        topk_ratio=_topk_ratio,
                        pad_mask_2d=pad_mask_2d,
                        head_ranked_rel_distances=_head_ranked_rel_distances,
                        rel_distance_topk_ratio=_rel_distance_topk_ratio,
                    )
                    del score_pre, score_post
                    capture[_layer_idx] = {
                        "head_sums": head_sums,
                        "head_counts": head_counts,
                        "head_key_sums": head_key_sums,
                        "head_dist_sums": head_dist_sums,
                        "head_dist_key_index_sums": head_dist_key_index_sums,
                        "head_rel_dist_topk_sums": head_rel_dist_topk_sums,
                        "head_union_sums": head_union_sums,
                        "head_union_key_index_sums": head_union_key_index_sums,
                        "batch_size": int(query_pre.size(0)),
                        "kv_len": kv_len,
                    }
                    _model._calib_post_mass_captured = True

                return _orig_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )

            return forward

        attn.forward = make_forward(
            layer_idx, orig_forward, topk_ratio, model, head_ranked_rel_distances, rel_distance_topk_ratio
        ).__get__(attn, type(attn))
        attn._calib_post_mass_decode_patched = True


def _first_answer_token_id(processor: Any, messages: List[Dict[str, Any]]) -> int:
    answer = str(messages[-1]["content"]).strip()
    token_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"empty answer tokenization: {answer!r}")
    return int(token_ids[0])


def _run_prefill_then_first_decode(
    model: torch.nn.Module,
    processor: Any,
    *,
    prompt_batch: Dict[str, torch.Tensor],
    messages: List[Dict[str, Any]],
    device: torch.device,
) -> None:
    prompt_inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in prompt_batch.items()
    }
    pad_mask_2d = prompt_inputs.get("attention_mask")
    if pad_mask_2d is not None and pad_mask_2d.dim() == 2:
        _set_post_mass_pad_mask(model, pad_mask_2d)
    else:
        _set_post_mass_pad_mask(model, None)

    prefill_out = model(**prompt_inputs, use_cache=True, return_dict=True)
    answer_id = _first_answer_token_id(processor, messages)
    decode_ids = torch.tensor([[answer_id]], dtype=torch.long, device=device)
    decode_attn = torch.cat(
        [prompt_inputs["attention_mask"], torch.ones_like(decode_ids, dtype=prompt_inputs["attention_mask"].dtype)],
        dim=-1,
    )
    if decode_attn.dim() == 2:
        _set_post_mass_pad_mask(model, decode_attn)
    model(
        input_ids=decode_ids,
        attention_mask=decode_attn,
        past_key_values=prefill_out.past_key_values,
        use_cache=True,
        return_dict=True,
    )


def _reset_decode_post_mass_state(model: torch.nn.Module) -> None:
    model._calib_post_mass_captured = False
    for attn in _iter_text_attention_modules(model):
        if getattr(attn, "_calib_post_mass_decode_patched", False):
            attn._calib_post_mass_pre_key_cache = None
            attn._calib_post_mass_post_key_cache = None


@dataclass
class Qwen3VLDecodePromptCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encodings = []
        for feat in features:
            enc = _vision_encode(
                self.processor,
                feat["messages"][:-1],
                add_generation_prompt=True,
            )
            encodings.append(enc)

        max_len = max(int(e["input_ids"].shape[1]) for e in encodings)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        input_ids = torch.full((len(encodings), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encodings), max_len), dtype=torch.long)
        mm_token_type_ids = torch.zeros((len(encodings), max_len), dtype=torch.long)

        pixel_values_videos = []
        video_grid_thw = []

        for i, enc in enumerate(encodings):
            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            mm = enc.get("mm_token_type_ids", torch.zeros_like(ids))
            length = int(ids.shape[0])

            input_ids[i, :length] = ids
            attention_mask[i, :length] = attn
            mm_token_type_ids[i, :length] = mm

            if "pixel_values_videos" in enc:
                pixel_values_videos.append(enc["pixel_values_videos"])
            if "video_grid_thw" in enc:
                video_grid_thw.append(enc["video_grid_thw"])

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        if pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
        if video_grid_thw:
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
        return batch


def parse_args():
    p = argparse.ArgumentParser(
        description="Post-RoPE softmax mass covered by pre-RoPE top-k keys"
    )
    p.add_argument("--model_path", type=str, default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--dataset", type=str, default="lmms-lab/TOMATO")
    p.add_argument("--dataset_fraction", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--layer_id", type=int, default=0)
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["last", "all", "decode"],
        help="last/all: prefill; decode: first decode step",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pre_qk_topk_ratio",
        "--sparse_topk_ratio",
        type=float,
        default=0.05,
        dest="pre_qk_topk_ratio",
        help="Pre-RoPE Q·K softmax: top ceil(ratio*n_valid) key indices per query row/head (0=disable)",
    )
    p.add_argument(
        "--distance_excel_dir",
        type=str,
        default=None,
        help="Distance-calibration excel input, e.g. .../all/excel/layer_00 (head_XX.xlsx)",
    )
    p.add_argument(
        "--print_mode",
        type=str,
        default="mean",
        choices=["full", "mean"],
        help="full: per-head lines + layer_mean; mean: layer_mean only",
    )
    p.add_argument(
        "--rel_distance_topk_ratio",
        "--distance_topk_ratio",
        type=float,
        default=0.05,
        dest="rel_distance_topk_ratio",
        help=(
            "Rel distance d=q_pos-k_pos: top ceil(ratio*n_valid) values by excel count; "
            "maps to key index k=q_pos-d (0=disable)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.min_pixels = resolve_min_pixels(args.max_pixels, args.min_pixels)

    frac = float(args.dataset_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"dataset_fraction must be in (0, 1], got {frac}")

    train_hf, _ = load_tomato_split(split="test", dataset_name=args.dataset, seed=args.seed)
    train_hf = train_hf.shuffle(seed=args.seed)
    n_take = max(1, int(len(train_hf) * frac))
    if args.limit is not None:
        n_take = min(n_take, args.limit)
    train_hf = train_hf.select(range(n_take))

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        trust_remote_code=True,
    )

    n_text_layers = len(_get_language_model(model).layers)
    if not (0 <= args.layer_id < n_text_layers):
        raise ValueError(f"layer_id={args.layer_id} out of range (num_layers={n_text_layers})")
    n_heads = model.config.text_config.num_attention_heads
    capture: dict = {}

    topk_ratio = float(args.pre_qk_topk_ratio)
    rel_distance_topk_ratio = float(args.rel_distance_topk_ratio)

    head_ranked_rel_distances: dict[int, list[int]] | None = None
    if args.distance_excel_dir and rel_distance_topk_ratio > 0:
        dist_excel_dir = Path(args.distance_excel_dir)
        head_ranked_rel_distances = load_head_ranked_rel_distances_from_excel(
            dist_excel_dir, n_heads=n_heads
        )
        if not head_ranked_rel_distances:
            raise FileNotFoundError(f"no head excel found under {dist_excel_dir}")
        print(
            f"[post-mass] rel_distance excel input: {dist_excel_dir} "
            f"top ceil({rel_distance_topk_ratio}*n_valid) rel_distance values/head by count "
            f"(key index k = q_pos - d)",
            flush=True,
        )
    elif args.distance_excel_dir and rel_distance_topk_ratio <= 0:
        print(
            f"[post-mass] rel_distance excel disabled (distance_topk_ratio=0), "
            f"skip {args.distance_excel_dir}",
            flush=True,
        )

    if topk_ratio > 0:
        print(
            f"[post-mass] pre-RoPE Q·K softmax top-k: ratio={topk_ratio} "
            f"(key_index count = ceil({topk_ratio}*n_valid))",
            flush=True,
        )
    else:
        print("[post-mass] pre-RoPE Q·K top-k disabled (pre_qk_topk_ratio=0)", flush=True)

    if topk_ratio <= 0 and head_ranked_rel_distances is None:
        raise ValueError(
            "need pre_qk_topk_ratio > 0 and/or distance_topk_ratio > 0 with --distance_excel_dir"
        )
    sample_layer_mean_acc = AllSamplesLayerMeanAccumulator(has_dist=rel_distance_topk_ratio > 0)

    dataset = TomatoSFTDataset(
        train_hf,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    device = next(model.parameters()).device

    if args.stage == "decode":
        install_decode_post_mass_hooks(
            model,
            capture,
            layer_id=args.layer_id,
            topk_ratio=topk_ratio,
            head_ranked_rel_distances=head_ranked_rel_distances,
            rel_distance_topk_ratio=rel_distance_topk_ratio,
        )
        collator = Qwen3VLDecodePromptCollator(processor=processor)
        batch_size = 1
        model.eval()
        print(
            f"[post-mass] stage=decode samples={len(dataset)} topk_ratio={topk_ratio} "
            f"layer_id={args.layer_id}",
            flush=True,
        )
        for sample_idx in tqdm(range(len(dataset)), desc="post-mass-decode"):
            capture.clear()
            _reset_decode_post_mass_state(model)
            batch = collator([dataset[sample_idx]])
            with torch.no_grad():
                _run_prefill_then_first_decode(
                    model,
                    processor,
                    prompt_batch=batch,
                    messages=dataset[sample_idx]["messages"],
                    device=device,
                )
            if args.layer_id not in capture:
                print(f"[post-mass] warn: no capture sample {sample_idx}, skip", flush=True)
                continue
            qk_mass, qk_n, dist_mass, rel_dist_n, matched_key_n, union_mass, union_n = (
                stats_per_head_from_capture(capture, args.layer_id)
            )
            sample_layer_mean_acc.add(
                qk_ratios=qk_mass,
                n_heads=n_heads,
                qk_key_index_counts=qk_n,
                dist_ratios=dist_mass,
                rel_dist_topk_counts=rel_dist_n,
                matched_key_index_counts=matched_key_n,
                union_ratios=union_mass,
                union_key_index_counts=union_n,
            )
            print_sample_head_coverage(
                sample_label=f"sample={sample_idx}",
                layer_id=args.layer_id,
                n_heads=n_heads,
                print_mode=args.print_mode,
                qk_ratios=qk_mass,
                qk_key_index_counts=qk_n,
                dist_ratios=dist_mass,
                rel_dist_topk_counts=rel_dist_n,
                matched_key_index_counts=matched_key_n,
                union_ratios=union_mass,
                union_key_index_counts=union_n,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        install_prefill_post_mass_hooks(
            model,
            capture,
            layer_id=args.layer_id,
            topk_ratio=topk_ratio,
            query_scope=args.stage,
            head_ranked_rel_distances=head_ranked_rel_distances,
            rel_distance_topk_ratio=rel_distance_topk_ratio,
        )
        _patch_text_model_early_stop(_get_language_model(model), args.layer_id)
        _patch_causal_lm_sparse_forward(model)
        collator = Qwen3VLDataCollator(processor=processor)
        batch_size = max(1, int(args.batch_size))
        model.eval()
        print(
            f"[post-mass] stage={args.stage} samples={len(dataset)} batch_size={batch_size} "
            f"topk_ratio={topk_ratio} layer_id={args.layer_id}",
            flush=True,
        )
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc="post-mass-prefill"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(dataset))
            capture.clear()
            batch = collator([dataset[i] for i in range(start, end)])
            forward_inputs = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch.items()
                if k != "labels"
            }
            forward_inputs["use_cache"] = False
            pad_mask_2d = forward_inputs.get("attention_mask")
            if pad_mask_2d is not None and pad_mask_2d.dim() == 2:
                _set_post_mass_pad_mask(model, pad_mask_2d)
            else:
                _set_post_mass_pad_mask(model, None)
            with torch.no_grad():
                model(**forward_inputs)
            if args.layer_id not in capture:
                print(f"[post-mass] warn: no capture batch {batch_idx}, skip", flush=True)
                continue
            if batch_size == 1:
                sample_label = f"sample={start}"
            else:
                sample_label = f"samples={start}-{end - 1}"
            qk_mass, qk_n, dist_mass, rel_dist_n, matched_key_n, union_mass, union_n = (
                stats_per_head_from_capture(capture, args.layer_id)
            )
            sample_layer_mean_acc.add(
                qk_ratios=qk_mass,
                n_heads=n_heads,
                qk_key_index_counts=qk_n,
                dist_ratios=dist_mass,
                rel_dist_topk_counts=rel_dist_n,
                matched_key_index_counts=matched_key_n,
                union_ratios=union_mass,
                union_key_index_counts=union_n,
            )
            print_sample_head_coverage(
                sample_label=sample_label,
                layer_id=args.layer_id,
                n_heads=n_heads,
                print_mode=args.print_mode,
                qk_ratios=qk_mass,
                qk_key_index_counts=qk_n,
                dist_ratios=dist_mass,
                rel_dist_topk_counts=rel_dist_n,
                matched_key_index_counts=matched_key_n,
                union_ratios=union_mass,
                union_key_index_counts=union_n,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_all_samples_summary(sample_layer_mean_acc, layer_id=args.layer_id)


if __name__ == "__main__":
    main()
