#!/usr/bin/env python3
"""Pack eval decode pre-softmax dump (per layer/head, key index) into rel-pos init .pt (by distance d)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def scores_row_to_distance_vector(scores_1d: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """d = q_pos - k_pos; first decode token has q_pos = kv_len - 1."""
    row = scores_1d.float().flatten()
    kv_len = row.numel()
    q_pos = kv_len - 1
    out = torch.zeros(num_buckets, dtype=torch.float32)
    for k in range(kv_len):
        d = q_pos - k
        if 0 <= d < num_buckets:
            out[d] = row[k]
    return out


def export_sample_dir(sample_dir: Path, output: Path, num_buckets: int) -> int:
    pat = re.compile(r"layer_(\d+)_head_(\d+)\.pt$")
    out: dict = {
        "meta": {
            "mode": "first_decode_token_to_previous_keys_by_distance",
            "source": str(sample_dir),
            "num_buckets": num_buckets,
        }
    }
    n = 0
    for f in sorted(sample_dir.glob("layer_*_head_*.pt")):
        m = pat.search(f.name)
        if not m:
            continue
        layer, head = int(m.group(1)), int(m.group(2))
        vec = scores_row_to_distance_vector(torch.load(f, map_location="cpu"), num_buckets)
        out[f"layer_{layer}.head_{head}"] = vec
        out[f"layer_{layer}.head_{head}.count"] = torch.ones(num_buckets)
        n += 1
    if n == 0:
        raise FileNotFoundError(f"No layer_*_head_*.pt under {sample_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", type=Path, required=True, help="e.g. /tmp/.../sample_00000")
    p.add_argument("--output", type=Path, default=Path("/tmp/baseline_relpos_scores.pt"))
    p.add_argument("--num-buckets", type=int, default=4096)
    args = p.parse_args()
    n = export_sample_dir(args.sample_dir, args.output, args.num_buckets)
    print(f"saved {n} heads -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
