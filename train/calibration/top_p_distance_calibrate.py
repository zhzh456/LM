#!/usr/bin/env python3
"""
Baseline prefill calibration: per layer/head, compare top-p key *relative distances*
between pre-RoPE QK scores (score1) and post-RoPE QK scores (score2).

For each sample, compare top-p key sets on pre-RoPE vs post-RoPE scores.
Use --query_scope last (prefill last token) or all (every valid query token).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))

from collator import Qwen3VLDataCollator
from dataset import (
    DEFAULT_MAX_PIXELS,
    TomatoSFTDataset,
    load_tomato_split,
    resolve_min_pixels,
)
from sparse_attention import qk_pre_softmax_scores
from patch_sparse_attn import _patch_causal_lm_sparse_forward, _patch_text_model_early_stop


def _get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    lm = getattr(model, "model", model)
    if hasattr(lm, "language_model"):
        return lm.language_model
    if hasattr(lm, "layers"):
        return lm
    raise AttributeError("Cannot find language model on Qwen3-VL model")


def _iter_text_attention_modules(model: torch.nn.Module):
    for layer in _get_language_model(model).layers:
        yield layer.self_attn


def _build_valid_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    q_positions: torch.Tensor,
    query_scope: str,
    device: torch.device,
    pad_mask_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bool (B, 1, Q, K): causal keys and query positions in scope."""
    q_idx = torch.arange(q_len, device=device, dtype=torch.long).view(1, q_len, 1)
    k_idx = torch.arange(kv_len, device=device, dtype=torch.long).view(1, 1, kv_len)
    causal = k_idx <= q_idx
    last_q = q_positions.to(device=device).view(batch_size, 1, 1)
    if query_scope == "last":
        q_ok = q_idx == last_q
    elif query_scope == "all":
        q_ok = q_idx <= last_q
    else:
        raise ValueError(f"unknown query_scope={query_scope!r}")
    valid = causal & q_ok
    if pad_mask_2d is not None:
        real = pad_mask_2d.to(device=device).bool()
        valid = valid & real.unsqueeze(1) & real.unsqueeze(2)
    return valid.unsqueeze(1)


def _nucleus_mask_bhqk(
    scores: torch.Tensor,
    top_p: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Top-p nucleus mask, shape matches scores (B, H, Q, K)."""
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~valid, neg_inf)
    probs = F.softmax(masked.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    n_below = (cumsum < top_p).sum(dim=-1, keepdim=True)
    ranks = torch.arange(scores.size(-1), device=scores.device).view(1, 1, 1, -1)
    in_sorted = ranks <= n_below
    nucleus = torch.zeros_like(scores, dtype=torch.bool)
    nucleus.scatter_(dim=-1, index=sorted_idx, src=in_sorted)
    return nucleus & valid


def _head_distance_counts(extra: torch.Tensor, rel_dist: torch.Tensor) -> dict[int, dict[int, int]]:
    """Count relative distances where extra is True. Tensors (B, H, Q, K)."""
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_heads = extra.size(1)
    for h in range(n_heads):
        d = rel_dist[:, h][extra[:, h]]
        if d.numel() == 0:
            continue
        d = d.long().reshape(-1)
        bc = torch.bincount(d, minlength=int(d.max().item()) + 1)
        for dist, cnt in enumerate(bc.tolist()):
            if cnt > 0:
                counts[h][dist] += cnt
    return {h: dict(dist_map) for h, dist_map in counts.items()}


def _accumulate_last_vectorized(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    q_positions: torch.Tensor,
    *,
    top_p: float,
    pad_mask_2d: torch.Tensor | None = None,
) -> dict[int, dict[int, int]]:
    """Fast path: only last query row per batch item."""
    batch_size = score_pre.size(0)
    n_heads = score_pre.size(1)
    kv_len = score_pre.size(-1)
    device = score_pre.device
    batch_idx = torch.arange(batch_size, device=device)
    last_q = q_positions.to(device=device)
    rows_pre = score_pre[batch_idx, :, last_q, :]
    rows_post = score_post[batch_idx, :, last_q, :]
    k_idx = torch.arange(kv_len, device=device).view(1, 1, kv_len)
    valid = k_idx <= last_q.view(batch_size, 1, 1)
    if pad_mask_2d is not None:
        valid = valid & pad_mask_2d.to(device=device).bool().view(batch_size, 1, kv_len)
    valid_bhqk = valid.unsqueeze(1)  # (B, 1, 1, K), broadcastable with (B, H, 1, K)
    nuc_pre = _nucleus_mask_bhqk(rows_pre.unsqueeze(2), top_p, valid_bhqk).squeeze(2)
    nuc_post = _nucleus_mask_bhqk(rows_post.unsqueeze(2), top_p, valid_bhqk).squeeze(2)
    rel_dist = (last_q.view(batch_size, 1, 1) - k_idx).expand(batch_size, n_heads, kv_len)
    extra = nuc_post & ~nuc_pre & valid.expand(batch_size, n_heads, kv_len)
    return _head_distance_counts(extra.unsqueeze(2), rel_dist.unsqueeze(2))


def accumulate_batch_extras(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    q_positions: torch.Tensor,
    *,
    top_p: float,
    query_scope: str,
    pad_mask_2d: torch.Tensor | None = None,
) -> dict[int, dict[int, int]]:
    """Per-head distance counts for one forward batch (GPU-vectorized)."""
    if query_scope == "last":
        return _accumulate_last_vectorized(
            score_pre, score_post, q_positions, top_p=top_p, pad_mask_2d=pad_mask_2d
        )

    batch_size, _, q_len, kv_len = score_pre.shape
    device = score_pre.device
    valid = _build_valid_mask(
        batch_size, q_len, kv_len, q_positions, query_scope, device, pad_mask_2d=pad_mask_2d
    )
    nuc_pre = _nucleus_mask_bhqk(score_pre, top_p, valid)
    nuc_post = _nucleus_mask_bhqk(score_post, top_p, valid)
    q_idx = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
    k_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
    rel_dist = (q_idx - k_idx).expand(batch_size, score_pre.size(1), q_len, kv_len)
    extra = nuc_post & ~nuc_pre & valid
    return _head_distance_counts(extra, rel_dist)


class ExtraDistanceAccumulator:
    def __init__(self, n_layers: int, n_heads: int) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        # counts[layer][head][distance] -> frequency across samples
        self.counts: dict[int, dict[int, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.n_samples = 0

    def add_extra(self, layer: int, head: int, extra: set[int]) -> None:
        for d in extra:
            self.counts[layer][head][d] += 1

    def to_serializable(self) -> dict:
        out: dict[str, dict] = {}
        for layer in range(self.n_layers):
            layer_obj: dict[str, dict] = {}
            for head in range(self.n_heads):
                dist_map = self.counts.get(layer, {}).get(head, {})
                if dist_map:
                    layer_obj[str(head)] = {str(d): int(c) for d, c in sorted(dist_map.items())}
            if layer_obj:
                out[str(layer)] = layer_obj
        return {
            "n_samples": self.n_samples,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "counts": out,
        }


def _q_positions_from_pad_mask(
    pad_mask_2d: torch.Tensor | None,
    *,
    batch_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    if pad_mask_2d is not None and pad_mask_2d.dim() == 2:
        return pad_mask_2d.long().sum(dim=-1).clamp(min=1).to(device=device) - 1
    return torch.full((batch_size,), kv_len - 1, dtype=torch.long, device=device)


def _set_calib_pad_mask(model: torch.nn.Module, pad_mask_2d: torch.Tensor | None) -> None:
    for attn in _iter_text_attention_modules(model):
        if getattr(attn, "_calib_capture_patched", False) or getattr(
            attn, "_calib_decode_capture_patched", False
        ):
            attn._calib_pad_mask_2d = pad_mask_2d


def install_score_capture_hooks(
    model: torch.nn.Module,
    capture: dict,
    *,
    layer_id: int = 0,
    top_p: float = 0.95,
    query_scope: str = "last",
) -> None:
    """Patch one text self-attn layer to record pre/post-RoPE scores and extras."""

    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        if getattr(attn, "_calib_capture_patched", False):
            continue

        orig_forward = attn.forward

        def make_forward(_layer_idx: int, _orig_forward, _top_p: float, _query_scope: str):
            def forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                # use_cache=False during calibration so K is full-sequence in one pass.
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
                        head_counts = accumulate_batch_extras(
                            score_pre,
                            score_post,
                            q_positions,
                            top_p=_top_p,
                            query_scope=_query_scope,
                            pad_mask_2d=pad_mask_2d,
                        )
                        del score_pre, score_post
                        capture[_layer_idx] = {
                            "head_counts": head_counts,
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

        attn.forward = make_forward(layer_idx, orig_forward, top_p, query_scope).__get__(
            attn, type(attn)
        )
        attn._calib_capture_patched = True


def process_capture(
    capture: dict,
    *,
    acc: ExtraDistanceAccumulator,
) -> int:
    """Merge per-forward head distance counts; returns number of batch rows processed."""
    n_rows = 0
    for layer_idx, data in capture.items():
        head_counts = data.get("head_counts", {})
        n_rows = max(n_rows, int(data.get("batch_size", 0)))
        for h, dist_counts in head_counts.items():
            for d, c in dist_counts.items():
                acc.counts[layer_idx][h][d] += int(c)
    return n_rows


def plot_layer_head_frequencies(
    acc: ExtraDistanceAccumulator,
    out_dir: Path,
    *,
    layer_id: int,
    query_scope: str = "last",
    max_distance_plot: int | None = None,
) -> None:
    """One PNG for the target layer: grid of head subplots (distance vs count)."""
    plot_root = out_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    head_maps = acc.counts.get(layer_id, {})
    if not head_maps:
        print(f"[calib] no counts for layer {layer_id}, skip plot", flush=True)
        return

    n_cols = 8
    n_rows = (acc.n_heads + n_cols - 1) // n_cols

    if max_distance_plot is None:
        max_d = 0
        for head_map in head_maps.values():
            if head_map:
                max_d = max(max_d, max(head_map.keys()))
        max_d = max(max_d, 0)
    else:
        max_d = max_distance_plot

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.0 * n_rows), squeeze=False)
    fig.suptitle(
        f"Layer {layer_id}: extra distances (set2 \\ set1) [{query_scope}]",
        fontsize=12,
    )

    for h in range(acc.n_heads):
        r, c = divmod(h, n_cols)
        ax = axes[r][c]
        dist_map = head_maps.get(h, {})
        xs = list(range(0, max_d + 1))
        ys = [dist_map.get(d, 0) for d in xs]
        ax.bar(xs, ys, width=0.8, color="#4C72B0")
        ax.set_title(f"head {h}", fontsize=8)
        ax.tick_params(labelsize=6)
        if r == n_rows - 1:
            ax.set_xlabel("distance", fontsize=7)
        if c == 0:
            ax.set_ylabel("count", fontsize=7)

    for h in range(acc.n_heads, n_rows * n_cols):
        r, c = divmod(h, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_root / f"layer_{layer_id:02d}_extra_distance_freq.png", dpi=120)
    plt.close(fig)


def export_layer_head_excel(
    acc: ExtraDistanceAccumulator,
    out_dir: Path,
    *,
    layer_id: int,
) -> None:
    """One .xlsx per head: columns distance, count (sorted by distance)."""
    import pandas as pd

    excel_root = out_dir / "excel" / f"layer_{layer_id:02d}"
    excel_root.mkdir(parents=True, exist_ok=True)

    head_maps = acc.counts.get(layer_id, {})
    n_written = 0
    for h in range(acc.n_heads):
        dist_map = head_maps.get(h, {})
        rows = [{"distance": int(d), "count": int(c)} for d, c in sorted(dist_map.items(), key=lambda x: x[1])]
        df = pd.DataFrame(rows, columns=["distance", "count"])
        out_path = excel_root / f"head_{h:02d}.xlsx"
        df.to_excel(out_path, index=False, sheet_name="extra_distances")
        n_written += 1

    print(f"[calib] excel -> {excel_root}/ ({n_written} files)", flush=True)


def save_calibration_results(
    acc: ExtraDistanceAccumulator,
    out_dir: Path,
    *,
    layer_id: int,
    meta: dict,
    max_distance_plot: int | None = None,
    label: str = "",
) -> str:
    """Write JSON + plot + excel. Returns output directory used."""
    save_root = out_dir / "checkpoints" / label if label else out_dir
    save_root.mkdir(parents=True, exist_ok=True)

    counts_path = save_root / "extra_distance_counts.json"
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **acc.to_serializable()}, f, indent=2)

    plot_layer_head_frequencies(
        acc,
        save_root,
        layer_id=layer_id,
        query_scope=str(meta.get("query_scope", "last")),
        max_distance_plot=max_distance_plot,
    )
    export_layer_head_excel(acc, save_root, layer_id=layer_id)

    tag = f" ({label})" if label else ""
    print(
        f"[calib] saved{tag} n_samples={acc.n_samples} -> {save_root}",
        flush=True,
    )
    return str(save_root)


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline top-p relative-distance calibration (pre-RoPE vs post-RoPE QK)"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct",
    )
    p.add_argument("--dataset", type=str, default="lmms-lab/TOMATO")
    p.add_argument("--output_dir", type=str, default="/tmp/qwen3vl-calibration-top-p")
    p.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of TOMATO to run (0,1], shuffled with --seed",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional hard cap on number of samples")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration forward (increase if GPU memory allows)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--layer_id",
        type=int,
        default=0,
        help="Text decoder layer to calibrate (default: 0 = first layer)",
    )
    p.add_argument(
        "--query_scope",
        type=str,
        default="last",
        choices=["last", "all"],
        help="last: only prefill last token; all: every valid query token in prefill",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Use eager for reliable score capture via patched forward",
    )
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Save JSON/plot/excel every N processed samples (0 = only at end)",
    )
    p.add_argument(
        "--max_distance_plot",
        type=int,
        default=None,
        help="Cap x-axis on plots; default = max distance seen per layer",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.min_pixels = resolve_min_pixels(args.max_pixels, args.min_pixels)
    if args.query_scope not in ("last", "all"):
        raise ValueError(f"query_scope must be 'last' or 'all', got {args.query_scope!r}")

    out_dir = Path(args.output_dir) / args.query_scope
    out_dir.mkdir(parents=True, exist_ok=True)

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
    acc = ExtraDistanceAccumulator(n_layers=n_text_layers, n_heads=n_heads)
    capture: dict = {}
    install_score_capture_hooks(
        model,
        capture,
        layer_id=args.layer_id,
        top_p=args.top_p,
        query_scope=args.query_scope,
    )
    _patch_text_model_early_stop(_get_language_model(model), args.layer_id)
    _patch_causal_lm_sparse_forward(model)

    dataset = TomatoSFTDataset(
        train_hf,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    collator = Qwen3VLDataCollator(processor=processor)
    device = next(model.parameters()).device

    meta_base = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_fraction": frac,
        "output_dir": str(out_dir),
        "top_p": args.top_p,
        "layer_id": args.layer_id,
        "num_frames": args.num_frames,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "save_interval": args.save_interval,
        "batch_size": args.batch_size,
        "query_scope": args.query_scope,
    }

    batch_size = max(1, int(args.batch_size))

    model.eval()
    print(
        f"[calib] samples={len(dataset)} batch_size={batch_size} top_p={args.top_p} "
        f"query_scope={args.query_scope} layer_id={args.layer_id} "
        f"(early-stop text decoder at layer {args.layer_id}) "
        f"heads={n_heads} max_pixels={args.max_pixels} attn={args.attn_implementation} "
        f"save_interval={args.save_interval} output={out_dir}",
        flush=True,
    )

    save_interval = max(0, int(args.save_interval))
    last_saved_at = 0

    n_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc="calibrate"):
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
            _set_calib_pad_mask(model, pad_mask_2d)
        else:
            _set_calib_pad_mask(model, None)
        with torch.no_grad():
            model(**forward_inputs)
        if args.layer_id not in capture:
            print(
                f"[calib] warn: no capture on layer {args.layer_id} batch {batch_idx} "
                f"(samples {start}-{end - 1}), skip",
                flush=True,
            )
            continue
        n_added = process_capture(capture, acc=acc)
        acc.n_samples += n_added
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_interval > 0 and acc.n_samples // save_interval > last_saved_at // save_interval:
            last_saved_at = (acc.n_samples // save_interval) * save_interval
            meta = {**meta_base, "n_samples_run": acc.n_samples, "checkpoint": True}
            save_calibration_results(
                acc,
                out_dir,
                layer_id=args.layer_id,
                meta=meta,
                max_distance_plot=args.max_distance_plot,
                label=f"samples_{acc.n_samples:05d}",
            )
            save_calibration_results(
                acc,
                out_dir,
                layer_id=args.layer_id,
                meta=meta,
                max_distance_plot=args.max_distance_plot,
            )

    meta = {**meta_base, "n_samples_run": acc.n_samples, "checkpoint": False}
    save_calibration_results(
        acc,
        out_dir,
        layer_id=args.layer_id,
        meta=meta,
        max_distance_plot=args.max_distance_plot,
    )
    print(f"[calib] final -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
