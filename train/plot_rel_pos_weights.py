#!/usr/bin/env python3
"""Plot trained per-head rel_pos_bias weights (first 511 buckets) as line charts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default="/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt",
    )
    p.add_argument("--out-dir", default="/tmp/Figure/2")
    p.add_argument(
        "--max-distance",
        type=int,
        default=6072,
        help="Plot d=0..max_distance inclusive (default≈train seq_len-1 for TOMATO max_pixels=12845056)",
    )
    p.add_argument("--max-layers", type=int, default=None, help="If set, only plot layer 0..max_layers-1")
    args = p.parse_args()
    n = args.max_distance + 1

    raw = torch.load(args.weights, map_location="cpu")
    try:
        from patch_sparse_attn import unpack_sparse_rel_pos_checkpoint

        meta, state = unpack_sparse_rel_pos_checkpoint(raw)
        if meta.get("train_layer_id") is not None:
            print(f"[plot] checkpoint train_layer_id={meta['train_layer_id']}")
    except Exception:
        state = raw if isinstance(raw, dict) else {}
    pat = re.compile(r"^layer_(\d+)\.head_(\d+)$")
    entries = []
    for key, vec in state.items():
        m = pat.match(key)
        if not m:
            continue
        entries.append((int(m.group(1)), int(m.group(2)), vec.float().flatten()))

    entries.sort(key=lambda x: (x[0], x[1]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = Path(args.weights).stem
    for layer, head, vec in entries:
        if args.max_layers is not None and layer >= args.max_layers:
            continue
        y = vec[:n].tolist()
        x = list(range(args.max_distance + 1))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, y, linewidth=0.9)
        ax.set_xlabel("relative distance d (q_pos - k_pos)")
        ax.set_ylabel("f(d)")
        ax.set_title(f"layer {layer} head {head} f(d) ({label}, d=0..{args.max_distance})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_head_{head:02d}.png", dpi=120)
        plt.close(fig)

    print(f"saved {len(entries)} figures to {out_dir}")


if __name__ == "__main__":
    main()
