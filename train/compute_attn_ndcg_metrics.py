#!/usr/bin/env python3
"""
NDCG@K: baseline 为真值，按 sparse pre-softmax 分数排序（first decode token）。
K = 10%, 20%, ..., 100%（10 档）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

K_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_pair(baseline_path: Path, sparse_path: Path) -> tuple[list[int], list[float], list[float]]:
    b = torch.load(baseline_path, map_location="cpu", weights_only=True).float().flatten()
    s = torch.load(sparse_path, map_location="cpu", weights_only=True).float().flatten()
    q = int(b.numel()) - 1
    ds, rel_b, rel_s = [], [], []
    for d in range(1, q + 1):
        k = q - d
        if k < 0:
            break
        ds.append(d)
        rel_b.append(float(b[k]))
        rel_s.append(float(s[k]))
    return ds, rel_b, rel_s


def _to_nonneg(rel: list[float]) -> list[float]:
    m = min(rel)
    return [r - m + 1e-8 for r in rel]


def dcg_at_k(rel_ordered: list[float], k: int) -> float:
    k = min(k, len(rel_ordered))
    return sum(rel_ordered[i] / math.log2(i + 2) for i in range(k))


def ndcg_sparse_vs_baseline(rel_baseline: list[float], rel_sparse_rank: list[float], k: int) -> float:
    """GT=baseline relevance; ranking=sparse score descending."""
    n = len(rel_baseline)
    if n == 0:
        return float("nan")
    rel = _to_nonneg(rel_baseline)
    order_pred = sorted(range(n), key=lambda i: rel_sparse_rank[i], reverse=True)
    order_ideal = sorted(range(n), key=lambda i: rel[i], reverse=True)
    rel_pred = [rel[i] for i in order_pred]
    rel_ideal = [rel[i] for i in order_ideal]
    dcg = dcg_at_k(rel_pred, k)
    idcg = dcg_at_k(rel_ideal, k)
    return 0.0 if idcg <= 0 else dcg / idcg


def iter_layer_heads(sample_dir: Path):
    pat = re.compile(r"layer_(\d+)_head_(\d+)\.pt$")
    for f in sorted(sample_dir.glob("layer_*_head_*.pt")):
        m = pat.search(f.name)
        if m:
            yield int(m.group(1)), int(m.group(2)), f


def process_sample(
    *,
    sample_id: str,
    baseline_dir: Path,
    sparse_dir: Path,
    out_dir: Path,
    max_layers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    b_root = baseline_dir / f"sample_{sample_id}"
    s_root = sparse_dir / f"sample_{sample_id}"

    rows = []
    agg: dict[tuple[int, float], list[float]] = {}

    for layer, head, b_path in iter_layer_heads(b_root):
        if layer >= max_layers:
            continue
        s_path = s_root / b_path.name
        if not s_path.exists():
            continue
        _, rel_b, rel_s = load_pair(b_path, s_path)
        n = len(rel_b)
        for frac in K_FRACS:
            k = max(1, int(round(n * frac)))
            ndcg = ndcg_sparse_vs_baseline(rel_b, rel_s, k)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "n_positions": n,
                    "k_pct": int(frac * 100),
                    "k": k,
                    "ndcg": ndcg,
                }
            )
            agg.setdefault((layer, frac), []).append(ndcg)

    with (out_dir / "ndcg_per_head.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    summary = {
        "sample": sample_id,
        "gt": "baseline",
        "rank": "sparse",
        "k_pct": [int(f * 100) for f in K_FRACS],
        "per_layer_mean": [],
    }
    for layer in sorted({L for L, _ in agg}):
        entry = {"layer": layer}
        for frac in K_FRACS:
            vals = agg[(layer, frac)]
            entry[f"ndcg@{int(frac * 100)}pct"] = sum(vals) / len(vals)
        summary["per_layer_mean"].append(entry)

    with (out_dir / "ndcg_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    for layer in sorted({L for L, _ in agg}):
        xs = [int(f * 100) for f in K_FRACS]
        ys = [sum(agg[(layer, f)]) / len(agg[(layer, f)]) for f in K_FRACS]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, "o-", color="#1f77b4", linewidth=1.5, markersize=6)
        ax.set_xlabel("K (% of valid positions)")
        ax.set_ylabel("NDCG@K")
        ax.set_title(f"sample {sample_id} layer {layer}  (GT=baseline, rank=sparse, mean over heads)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"ndcg_layer_{layer:02d}.png", dpi=120)
        plt.close(fig)

    print(f"sample {sample_id}: {len(rows)} rows -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--sparse-dir", required=True)
    p.add_argument("--out-dir", default=None, help="Single sample metrics dir")
    p.add_argument("--sample-id", default="00000")
    p.add_argument("--out-root", default=None, help="Legacy: metric/5 metric/6")
    p.add_argument("--max-layers", type=int, default=2)
    args = p.parse_args()

    if args.out_dir:
        pairs = [(args.sample_id, Path(args.out_dir))]
    else:
        root = Path(args.out_root or "/tmp/Figure/metric")
        pairs = [("00000", root / "5"), ("00001", root / "6")]

    for sample_id, out_dir in pairs:
        process_sample(
            sample_id=sample_id,
            baseline_dir=Path(args.baseline_dir),
            sparse_dir=Path(args.sparse_dir),
            out_dir=out_dir,
            max_layers=args.max_layers,
        )


if __name__ == "__main__":
    main()
