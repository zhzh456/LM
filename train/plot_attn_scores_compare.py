#!/usr/bin/env python3
"""Plot pre-softmax decode attention: baseline vs sparse, one sample per output dir."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _load_curve_by_distance(path: Path, max_distance: int = 511) -> tuple[list[int], list[float]]:
    """Map decode scores over key index k to relative distance d = q - k (q = last key index)."""
    scores = torch.load(path, map_location="cpu").float()
    if scores.dim() == 2:
        scores = scores[-1, :]
    q = int(scores.numel()) - 1
    n = min(max_distance, q + 1)
    x, y = [], []
    for d in range(1, n):
        k = q - d
        if k < 0:
            break
        x.append(d)
        y.append(float(scores[k]))
    return x, y


def _iter_layer_head_files(sample_dir: Path):
    pat = re.compile(r"layer_(\d+)_head_(\d+)\.pt$")
    for f in sorted(sample_dir.glob("layer_*_head_*.pt")):
        m = pat.search(f.name)
        if m:
            yield int(m.group(1)), int(m.group(2)), f


def _plot_sample(
    *,
    sample_id: str,
    baseline_dir: Path,
    sparse_dir: Path,
    out_dir: Path,
    max_layers: int,
    max_distance: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    b_dir = baseline_dir / f"sample_{sample_id}"
    s_dir = sparse_dir / f"sample_{sample_id}"
    count = 0
    for layer, head, _ in _iter_layer_head_files(b_dir):
        if layer >= max_layers:
            continue
        name = f"layer_{layer:02d}_head_{head:02d}.png"
        paths = {
            "baseline": b_dir / f"layer_{layer:02d}_head_{head:02d}.pt",
            "sparse": s_dir / f"layer_{layer:02d}_head_{head:02d}.pt",
        }
        fig, ax = plt.subplots(figsize=(10, 4))
        for label, path in paths.items():
            if not path.exists():
                continue
            x, y = _load_curve_by_distance(path, max_distance=max_distance)
            ax.plot(x, y, label=label, linewidth=0.8, alpha=0.9)

        ax.set_xlabel("relative distance d (q_pos - k_pos)")
        ax.set_ylabel("pre-softmax score")
        ax.set_title(f"sample {sample_id} layer {layer} head {head} (first decode token)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
        count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--sparse-dir", required=True)
    p.add_argument("--out-root", default=None, help="Legacy: out_root/sample{N}/")
    p.add_argument("--out-dir", default=None, help="Single sample output dir (overrides out-root pair)")
    p.add_argument("--sample-id", default="00000", help="Dump subfolder sample_XXXXX")
    p.add_argument("--max-layers", type=int, default=2)
    p.add_argument("--max-distance", type=int, default=511)
    p.add_argument("--sample1-subdir", default="1")
    p.add_argument("--sample2-subdir", default="2")
    args = p.parse_args()

    baseline_root = Path(args.baseline_dir)
    sparse_root = Path(args.sparse_dir)

    if args.out_dir:
        pairs = [(args.sample_id, Path(args.out_dir))]
    else:
        out_root = Path(args.out_root or "/tmp/Figure/1")
        pairs = [
            ("00000", out_root / args.sample1_subdir),
            ("00001", out_root / args.sample2_subdir),
        ]

    for sample_id, out_dir in pairs:
        n = _plot_sample(
            sample_id=sample_id,
            baseline_dir=baseline_root,
            sparse_dir=sparse_root,
            out_dir=out_dir,
            max_layers=args.max_layers,
            max_distance=args.max_distance,
        )
        print(f"sample {sample_id}: saved {n} figures to {out_dir}")


if __name__ == "__main__":
    main()
