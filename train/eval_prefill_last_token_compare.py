#!/usr/bin/env python3
"""
Single-sample compare: prefill last-token pre-softmax scores (sparse layer only).

- GT (baseline): RoPE(Q)·RoPE(K)/sqrt(d)
- Predicted (sparse): f(d)·Q_pre·K_pre/sqrt(d)

Per head: one figure with 3 subplots (NDCG, pre-softmax attention, f(d) weights).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from collator import Qwen3VLDataCollator
from dataset import TomatoSFTDataset, load_tomato_split, resolve_min_pixels
from patch_sparse_attn import (
    _assemble_pre_rope_keys,
    _get_stacked_rel_pos_bias,
    _iter_text_attention_modules,
    load_sparse_rel_pos_checkpoint,
    patch_model_for_sparse_eval,
    reset_sparse_pre_rope_key_caches,
    unpack_sparse_rel_pos_checkpoint,
)
from sparse_attention import _sparse_pre_softmax_scores, qk_pre_softmax_scores

K_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _to_nonneg(rel: list[float]) -> list[float]:
    m = min(rel)
    return [r - m + 1e-8 for r in rel]


def dcg_at_k(rel_ordered: list[float], k: int) -> float:
    k = min(k, len(rel_ordered))
    return sum(rel_ordered[i] / math.log2(i + 2) for i in range(k))


def ndcg_sparse_vs_baseline(rel_baseline: list[float], rel_sparse_rank: list[float], k: int) -> float:
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


def load_head_weights(sparse_weights: Path, layer_id: int, n_heads: int) -> list[torch.Tensor]:
    raw = torch.load(sparse_weights, map_location="cpu", weights_only=False)
    try:
        _, state = unpack_sparse_rel_pos_checkpoint(raw)
    except Exception:
        state = raw if isinstance(raw, dict) else {}
    vecs: list[torch.Tensor] = []
    for h in range(n_heads):
        key = f"layer_{layer_id}.head_{h}"
        if key not in state:
            raise KeyError(f"missing {key} in {sparse_weights}")
        vecs.append(state[key].float().flatten())
    return vecs


def scores_row_to_distance_curve(row: torch.Tensor) -> tuple[list[int], list[float]]:
    """Last query row over keys -> (d, score) for d = q_pos - k_pos, d>=1 (ascending d)."""
    vec = row.float().flatten()
    q_pos = int(vec.numel()) - 1
    ds, ys = [], []
    for d in range(1, q_pos + 1):
        k = q_pos - d
        ds.append(d)
        ys.append(float(vec[k]))
    return ds, ys


def capture_prefill_last_token_scores(
    model: torch.nn.Module,
    inputs: dict,
    *,
    layer_id: int = 35,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return baseline/sparse last-query rows (num_heads, kv_len) and seq_len."""
    captured: dict[str, torch.Tensor] = {}

    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        if not getattr(attn, "_sparse_forward_patched", False):
            raise RuntimeError(f"layer {layer_id} is not sparse-patched")

        orig_forward = attn.forward

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
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_pre_rope = query_states
            key_pre_rope_step = key_states

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )

            q_len = query_states.size(-2)
            kv_len = key_states.size(-2)
            track_cache = getattr(self, "sparse_decode_only", False)
            key_pre_rope = _assemble_pre_rope_keys(
                self, key_pre_rope_step, track_cache=track_cache
            )
            prev_kv = int(captured.get("kv_len", -1)) if "kv_len" in captured else -1
            if kv_len >= prev_kv:
                rel_bias = _get_stacked_rel_pos_bias(self)
                b = qk_pre_softmax_scores(
                    self, query_states, key_states, attention_mask, self.scaling
                )
                s = _sparse_pre_softmax_scores(
                    self,
                    query_pre_rope,
                    key_pre_rope,
                    rel_bias,
                    query_states.size(0),
                    q_len,
                    kv_len,
                    self.scaling,
                    attention_mask=attention_mask,
                    dtype=query_states.dtype,
                )
                captured["baseline"] = b[0, :, -1, :].detach().float().cpu()
                captured["sparse"] = s[0, :, -1, :].detach().float().cpu()
                captured["seq_len"] = torch.tensor(kv_len)
                captured["kv_len"] = kv_len

            return orig_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        attn.forward = forward.__get__(attn, type(attn))
        break

    device = next(model.parameters()).device
    reset_sparse_pre_rope_key_caches(model)
    model.eval()
    with torch.no_grad():
        model(**{k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()})

    if "baseline" not in captured:
        raise RuntimeError("No prefill capture at sparse layer")

    seq_len = int(captured["seq_len"].item())
    return captured["baseline"], captured["sparse"], seq_len


def plot_and_metrics(
    *,
    baseline: torch.Tensor,
    sparse: torch.Tensor,
    out_dir: Path,
    sample_idx: int,
    layer_id: int,
    sparse_weights: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_b = out_dir / "dump_baseline"
    dump_s = out_dir / "dump_sparse"
    dump_b.mkdir(exist_ok=True)
    dump_s.mkdir(exist_ok=True)

    n_heads = baseline.size(0)
    seq_len = int(baseline.size(-1))
    head_weights = load_head_weights(sparse_weights, layer_id, n_heads)
    max_d = min(seq_len - 1, min(w.numel() for w in head_weights) - 1)
    xs_ndcg = [int(f * 100) for f in K_FRACS]
    rows = []

    for h in range(n_heads):
        b_row = baseline[h]
        s_row = sparse[h]
        torch.save(b_row, dump_b / f"layer_{layer_id:02d}_head_{h:02d}.pt")
        torch.save(s_row, dump_s / f"layer_{layer_id:02d}_head_{h:02d}.pt")

        ds, yb = scores_row_to_distance_curve(b_row)
        _, ys = scores_row_to_distance_curve(s_row)
        q_pos = int(b_row.numel()) - 1

        n = len(yb)
        ndcg_curve: list[float] = []
        for frac in K_FRACS:
            k = max(1, int(round(n * frac)))
            ndcg = ndcg_sparse_vs_baseline(yb, ys, k)
            ndcg_curve.append(ndcg)
            rows.append(
                {
                    "layer": layer_id,
                    "head": h,
                    "n_positions": n,
                    "k_pct": int(frac * 100),
                    "k": k,
                    "ndcg": ndcg,
                }
            )

        w_d = list(range(max_d + 1))
        w_y = head_weights[h][: max_d + 1].tolist()

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        fig.suptitle(
            f"sample {sample_idx} layer {layer_id} head {h} "
            f"(prefill last token, seq_len={q_pos + 1})",
            fontsize=11,
        )

        ax0 = axes[0]
        ax0.plot(xs_ndcg, ndcg_curve, "o-", linewidth=1.5, markersize=5, color="#2ca02c")
        ax0.set_xlabel("K (% of key positions)")
        ax0.set_ylabel("NDCG@K")
        ax0.set_title("NDCG (GT=baseline)")
        ax0.set_ylim(0, 1.05)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1]
        ax1.plot(ds, yb, label="baseline (RoPE QK)", linewidth=0.9, alpha=0.9)
        ax1.plot(ds, ys, label="sparse f(d)·Q_pre·K_pre/√d", linewidth=0.9, alpha=0.9)
        ax1.set_xlabel("relative distance d")
        ax1.set_ylabel("pre-softmax score")
        ax1.set_title("attention (last query)")
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[2]
        ax2.plot(w_d, w_y, linewidth=0.9, color="#9467bd")
        ax2.set_xlabel("relative distance d")
        ax2.set_ylabel("f(d)")
        ax2.set_title("rel_pos weights")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer_id:02d}_head_{h:02d}.png", dpi=120)
        plt.close(fig)

    with (out_dir / "ndcg_per_head.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg: dict[float, list[float]] = {frac: [] for frac in K_FRACS}
    for r in rows:
        agg[r["k_pct"] / 100.0].append(r["ndcg"])

    summary = {
        "sample": sample_idx,
        "layer": layer_id,
        "gt": "baseline_rope_qk",
        "rank": "sparse_pre_rope_qk_f_d",
        "prefill_last_token": True,
        "k_pct": [int(f * 100) for f in K_FRACS],
        "mean_ndcg": {f"ndcg@{int(f*100)}pct": sum(agg[f]) / len(agg[f]) for f in K_FRACS},
    }
    with (out_dir / "ndcg_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"seq_len={seq_len} heads={n_heads}")
    print(f"NDCG@10%={summary['mean_ndcg']['ndcg@10pct']:.4f} NDCG@100%={summary['mean_ndcg']['ndcg@100pct']:.4f}")
    print(f"saved {n_heads} combined figures + metrics -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument(
        "--sparse_weights",
        default="/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt",
    )
    p.add_argument("--out-dir", default="/tmp/Figure/prefill_last_token_layer35")
    p.add_argument("--sample-index", type=int, default=None, help="Single sample (overrides --num-samples)")
    p.add_argument("--num-samples", type=int, default=2, help="Plot samples 0..num_samples-1")
    p.add_argument("--layer-id", type=int, default=35)
    p.add_argument("--max_pixels", type=int, default=12845056)
    p.add_argument("--num_frames", type=int, default=16)
    args = p.parse_args()

    os.environ.setdefault("http_proxy", "http://10.229.18.27:8412")
    os.environ.setdefault("https_proxy", "http://10.229.18.27:8412")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.sample_index is not None:
        sample_indices = [args.sample_index]
    else:
        sample_indices = list(range(args.num_samples))

    min_pixels = resolve_min_pixels(args.max_pixels)
    train_hf, _ = load_tomato_split(split="test", limit=max(sample_indices) + 1)
    ds = TomatoSFTDataset(
        train_hf,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        min_pixels=min_pixels,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=min_pixels,
        trust_remote_code=True,
    )
    patch_model_for_sparse_eval(model, layer_id=args.layer_id, num_buckets=16384)
    load_sparse_rel_pos_checkpoint(model, args.sparse_weights, expected_layer_id=args.layer_id)

    collator = Qwen3VLDataCollator(processor=processor)
    out_root = Path(args.out_dir)

    for sample_idx in sample_indices:
        sample = ds[sample_idx]
        inputs = collator([sample])
        inputs.pop("labels", None)

        baseline, sparse, _ = capture_prefill_last_token_scores(
            model, inputs, layer_id=args.layer_id
        )
        plot_and_metrics(
            baseline=baseline,
            sparse=sparse,
            out_dir=out_root / f"sample_{sample_idx:02d}",
            sample_idx=sample_idx,
            layer_id=args.layer_id,
            sparse_weights=Path(args.sparse_weights),
        )


if __name__ == "__main__":
    main()
