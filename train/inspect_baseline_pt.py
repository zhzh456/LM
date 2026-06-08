#!/usr/bin/env python3
import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect baseline_relpos_scores.pt quickly")
    parser.add_argument(
        "--path",
        default="/tmp/baseline_relpos_scores.pt",
        help="Path to .pt file",
    )
    parser.add_argument(
        "--show_keys",
        type=int,
        default=10,
        help="How many keys to print",
    )
    args = parser.parse_args()

    obj = torch.load(args.path, map_location="cpu")
    import pdb

    pdb.set_trace()
    if not isinstance(obj, dict):
        print(f"type={type(obj)}")
        return
    keys = list(obj.keys())
    print(f"path={args.path}")
    print(f"total_keys={len(keys)}")
    print(f"first_{args.show_keys}_keys:")
    for k in keys[: args.show_keys]:
        print(f"  {k}")

    sample_key = "layer_0.head_0"
    if sample_key in obj and torch.is_tensor(obj[sample_key]):
        t = obj[sample_key].flatten()
        n = min(8, t.numel())
        vals = ", ".join(f"{float(x):.6f}" for x in t[:n])
        print(f"{sample_key}.shape={tuple(obj[sample_key].shape)}")
        print(f"{sample_key}.first_{n}={vals}")


if __name__ == "__main__":
    main()
