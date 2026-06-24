#!/usr/bin/env python3
"""
STEP 1: Layer sensitivity profiling.

For each text layer l, perturb only that layer with retention ratio r
(default 0.5), unless --all is set (all listed layers use r jointly).

Sparsity = post-RoPE Q·K logits top-k
(keep ceil(r * n_valid) keys per query/head; same r for all heads in the layer).
Top-k applies on prefill only; decode uses full post-RoPE attention.

Other layers stay dense. Record Acc_l(r); drop vs Acc_full is optional (--measure_acc_full).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys


def _ensure_hf_proxy() -> None:
    """与 examples/models/qwen3vl_sparse_attn.sh 一致，在加载 HF 资源前设置代理。"""
    proxy = "http://10.229.18.27:8412"
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


_ensure_hf_proxy()
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))

from budget_opt.post_rope_topk import (  # noqa: E402
    clear_post_rope_topk_layers,
    install_post_rope_topk_uniform_layers,
    num_text_layers,
    set_post_rope_topk_single_layer,
)


DEFAULT_RATIOS = [0.5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEP 1: per-layer sensitivity profiling")
    p.add_argument("--model_path", type=str, default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--tasks", type=str, default="tomato")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="/tmp/budget-opt/step1")
    p.add_argument(
        "--measure_acc_full",
        action="store_true",
        help="also run dense baseline Acc_full before per-layer profiling",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="apply --ratios to every layer at once (one eval per ratio), not one layer at a time",
    )
    p.add_argument("--layers", type=str, default=None, help="e.g. 0,1,2 or 0-31")
    p.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="comma-separated retention ratios, default 0.5",
    )
    p.add_argument("--max_pixels", type=int, default=12845056)
    p.add_argument("--max_num_frames", type=int, default=88)
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="post-RoPE top-k 需要显式 Q·K 分数，须用 eager（flash 无法动态选 KV）",
    )
    return p.parse_args()


def parse_layer_list(spec: str | None, n_layers: int) -> list[int]:
    if spec is None:
        return list(range(n_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({i for i in out if 0 <= i < n_layers})


def parse_ratio_list(spec: str | None) -> list[float]:
    if spec is None:
        return list(DEFAULT_RATIOS)
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def extract_metric(results: dict[str, Any], task: str, metric_prefix: str) -> float | None:
    task_res = results.get("results", {}).get(task, {})
    for key, value in task_res.items():
        if key.split(",")[0] == metric_prefix:
            return float(value)
    return None


def metric_name_for_task(task: str) -> str:
    if task == "tomato":
        return "tomato_score"
    return f"{task}_acc"


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _ratio_key(ratio: float) -> str:
    return f"{ratio:.1f}"


def run_eval(lm, hf_model, tasks: list[str], limit: int | None, batch_size: int) -> dict[str, Any]:
    """Run lmms_eval; restore lm._model after clean() deletes nn.Module attrs."""
    from lmms_eval import evaluator

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
        log_samples=False,
    )
    # simple_evaluate -> evaluate() calls lm.clean(), which delattr's self._model.
    lm._model = hf_model
    return results


def build_layer_table_rows(
    layer_entry: dict[str, Any],
    ratios: list[float],
    acc_full: float | None,
) -> list[dict[str, float]]:
    acc_map = layer_entry.get("acc_by_ratio", {})
    drop_map = layer_entry.get("drop_by_ratio", {})
    rows: list[dict[str, float]] = []
    for ratio in ratios:
        key = _ratio_key(ratio)
        if key in acc_map:
            acc = float(acc_map[key])
        elif acc_full is not None and ratio >= 1.0:
            acc = acc_full
        else:
            acc = math.nan
        if key in drop_map:
            drop = float(drop_map[key])
        elif acc_full is not None and math.isfinite(acc):
            drop = acc_full - acc
        else:
            drop = math.nan
        rows.append({"r": ratio, "acc": acc, "drop": drop})
    return rows


def _fmt_float(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return ""
    return f"{x:.6f}"


def export_layer_tables(
    store: dict[str, Any],
    output_dir: Path,
    *,
    layer_ids: list[int],
    ratios: list[float],
    acc_full: float | None,
) -> dict[str, list[dict[str, float]]]:
    """Write per-layer CSV tables and return tables dict for JSON."""
    table_dir = output_dir / "layer_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    include_drop = acc_full is not None
    csv_fields = ["r", "acc", "drop"] if include_drop else ["r", "acc"]

    tables: dict[str, list[dict[str, float]]] = {}
    all_rows: list[dict[str, Any]] = []

    acc_matrix_path = output_dir / "step1_acc_matrix.csv"
    drop_matrix_path = output_dir / "step1_drop_matrix.csv"
    acc_header = ["layer_id"] + [_ratio_key(r) for r in ratios]
    acc_matrix_rows: list[list[Any]] = []
    drop_matrix_rows: list[list[Any]] = []

    for layer_id in layer_ids:
        layer_key = str(layer_id)
        layer_entry = store.get("layers", {}).get(layer_key, {})
        rows = build_layer_table_rows(layer_entry, ratios, acc_full)
        tables[layer_key] = rows

        layer_csv = table_dir / f"layer_{layer_id:02d}.csv"
        with layer_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in rows:
                out = {"r": f"{row['r']:.1f}", "acc": _fmt_float(row["acc"])}
                if include_drop:
                    out["drop"] = _fmt_float(row["drop"])
                writer.writerow(out)

        acc_row = [layer_id] + [_fmt_float(r["acc"]) for r in rows]
        acc_matrix_rows.append(acc_row)
        if include_drop:
            drop_row = [layer_id] + [_fmt_float(r["drop"]) for r in rows]
            drop_matrix_rows.append(drop_row)

        for row in rows:
            item: dict[str, Any] = {"layer_id": layer_id, "r": row["r"], "acc": row["acc"]}
            if include_drop:
                item["drop"] = row["drop"]
            all_rows.append(item)

    with (output_dir / "step1_all_layers.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_id"] + csv_fields)
        writer.writeheader()
        for row in all_rows:
            out = {
                "layer_id": row["layer_id"],
                "r": f"{row['r']:.1f}",
                "acc": _fmt_float(row["acc"]),
            }
            if include_drop:
                out["drop"] = _fmt_float(row["drop"])
            writer.writerow(out)

    with acc_matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(acc_header)
        writer.writerows(acc_matrix_rows)

    if include_drop:
        with drop_matrix_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(acc_header)
            writer.writerows(drop_matrix_rows)

    return tables


def print_layer_table(layer_id: int, rows: list[dict[str, float]], acc_full: float | None) -> None:
    if acc_full is not None:
        print(f"\n[step1] layer {layer_id}  (Acc_full={acc_full:.6f})", flush=True)
        print("  r    acc      drop", flush=True)
        print("  ---- -------- --------", flush=True)
        for row in rows:
            acc_s = _fmt_float(row["acc"]) or "nan"
            drop_s = _fmt_float(row["drop"]) or "nan"
            print(f"  {row['r']:.1f}  {acc_s:>8}  {drop_s:>8}", flush=True)
    else:
        print(f"\n[step1] layer {layer_id}", flush=True)
        print("  r    acc", flush=True)
        print("  ---- --------", flush=True)
        for row in rows:
            acc_s = _fmt_float(row["acc"]) or "nan"
            print(f"  {row['r']:.1f}  {acc_s:>8}", flush=True)


def print_all_layers_table(rows: list[dict[str, float]], acc_full: float | None) -> None:
    if acc_full is not None:
        print(f"\n[step1] all layers  (Acc_full={acc_full:.6f})", flush=True)
        print("  r    acc      drop", flush=True)
        print("  ---- -------- --------", flush=True)
        for row in rows:
            acc_s = _fmt_float(row["acc"]) or "nan"
            drop_s = _fmt_float(row["drop"]) or "nan"
            print(f"  {row['r']:.1f}  {acc_s:>8}  {drop_s:>8}", flush=True)
    else:
        print("\n[step1] all layers (joint)", flush=True)
        print("  r    acc", flush=True)
        print("  ---- --------", flush=True)
        for row in rows:
            acc_s = _fmt_float(row["acc"]) or "nan"
            print(f"  {row['r']:.1f}  {acc_s:>8}", flush=True)


def finalize_all_layers_outputs(
    store: dict[str, Any],
    output_dir: Path,
    *,
    ratios: list[float],
    acc_full: float | None,
    results_path: Path,
) -> None:
    entry = store.get("all", {})
    rows = build_layer_table_rows(entry, ratios, acc_full)
    include_drop = acc_full is not None
    csv_fields = ["r", "acc", "drop"] if include_drop else ["r", "acc"]

    with (output_dir / "step1_all_layers_joint.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            out = {"r": f"{row['r']:.1f}", "acc": _fmt_float(row["acc"])}
            if include_drop:
                out["drop"] = _fmt_float(row["drop"])
            writer.writerow(out)

    store["tables"] = {"all": rows}
    if acc_full is not None:
        store["acc_full"] = acc_full
    save_results(results_path, store)
    print_all_layers_table(rows, acc_full)
    print(f"\n[step1] joint -> {output_dir / 'step1_all_layers_joint.csv'}", flush=True)
    print(f"[step1] json   -> {results_path}", flush=True)


def finalize_step1_outputs(
    store: dict[str, Any],
    output_dir: Path,
    *,
    layer_ids: list[int],
    ratios: list[float],
    acc_full: float | None,
    results_path: Path,
) -> None:
    tables = export_layer_tables(
        store,
        output_dir,
        layer_ids=layer_ids,
        ratios=ratios,
        acc_full=acc_full,
    )
    if acc_full is not None:
        store["acc_full"] = acc_full
    store["tables"] = tables
    save_results(results_path, store)

    for layer_id in layer_ids:
        print_layer_table(layer_id, tables[str(layer_id)], acc_full)

    print(
        f"\n[step1] tables -> {output_dir / 'layer_tables'}/layer_XX.csv",
        flush=True,
    )
    print(f"[step1] matrix -> {output_dir / 'step1_acc_matrix.csv'}", flush=True)
    if acc_full is not None:
        print(f"[step1] drop   -> {output_dir / 'step1_drop_matrix.csv'}", flush=True)
    print(f"[step1] json    -> {results_path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_path = output_dir / "step1_sensitivity.json"

    from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

    lm = Qwen3_VL(
        pretrained=args.model_path,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
        max_pixels=args.max_pixels,
        max_num_frames=args.max_num_frames,
        interleave_visuals=False,
    )
    hf_model = lm._model

    n_layers = num_text_layers(hf_model)
    layer_ids = parse_layer_list(args.layers, n_layers)
    ratios = parse_ratio_list(args.ratios)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    primary_task = tasks[0]
    metric = metric_name_for_task(primary_task)

    store: dict[str, Any] = {
        "meta": {
            "step": 1,
            "mode": "all_layers_joint" if args.all else "single_layer",
            "model_path": args.model_path,
            "tasks": tasks,
            "metric": metric,
            "n_text_layers": n_layers,
            "layers_profiled": layer_ids,
            "ratios": ratios,
            "sparsity": "post_rope_qk_topk_prefill_only",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        "layers": {},
    }
    if args.all:
        store["all"] = {"acc_by_ratio": {}, "drop_by_ratio": {}}

    acc_full: float | None = None
    if args.measure_acc_full:
        print("[step1] measuring Acc_full (all layers dense)", flush=True)
        clear_post_rope_topk_layers(hf_model)
        full_results = run_eval(lm, hf_model, tasks, args.limit, args.batch_size)
        acc_full = extract_metric(full_results, primary_task, metric)
        if acc_full is None:
            raise RuntimeError(f"metric {metric!r} not found in results: {full_results.get('results', {})}")
        store["acc_full"] = acc_full
        save_results(results_path, store)
        print(f"[step1] Acc_full = {acc_full:.6f}", flush=True)

    if args.all:
        all_entry = store["all"]
        for ratio in ratios:
            if ratio >= 1.0 and acc_full is not None:
                acc = acc_full
                print(f"[step1] all layers r={ratio:.1f} -> Acc={acc:.6f} (dense)", flush=True)
            else:
                clear_post_rope_topk_layers(hf_model)
                install_post_rope_topk_uniform_layers(hf_model, layer_ids=layer_ids, ratio=ratio)
                layer_desc = f"{layer_ids[0]}-{layer_ids[-1]}" if len(layer_ids) > 1 else str(layer_ids[0])
                print(
                    f"[step1] eval ALL layers [{layer_desc}] n={len(layer_ids)} "
                    f"r={ratio:.1f} (post-RoPE top-k prefill)",
                    flush=True,
                )
                eval_results = run_eval(lm, hf_model, tasks, args.limit, args.batch_size)
                acc = extract_metric(eval_results, primary_task, metric)
                if acc is None:
                    raise RuntimeError(
                        f"metric {metric!r} missing for all_layers r={ratio}: "
                        f"{eval_results.get('results', {})}"
                    )
                print(f"[step1] all layers r={ratio:.1f} -> Acc={acc:.6f}", flush=True)

            ratio_key = _ratio_key(ratio)
            all_entry["acc_by_ratio"][ratio_key] = acc
            if acc_full is not None:
                all_entry["drop_by_ratio"][ratio_key] = acc_full - acc
            store["all"] = all_entry
            save_results(results_path, store)

        clear_post_rope_topk_layers(hf_model)
        finalize_all_layers_outputs(
            store,
            output_dir,
            ratios=ratios,
            acc_full=acc_full,
            results_path=results_path,
        )
        return

    for layer_id in layer_ids:
        layer_key = str(layer_id)
        layer_entry = store["layers"].setdefault(
            layer_key,
            {"acc_by_ratio": {}, "drop_by_ratio": {}},
        )
        for ratio in ratios:
            if ratio >= 1.0 and acc_full is not None:
                acc = acc_full
                print(f"[step1] layer={layer_id} r={ratio:.1f} -> Acc={acc:.6f} (dense)", flush=True)
            else:
                clear_post_rope_topk_layers(hf_model)
                set_post_rope_topk_single_layer(hf_model, layer_id=layer_id, ratio=ratio)
                print(f"[step1] eval layer={layer_id} r={ratio:.1f} (post-RoPE top-k)", flush=True)
                eval_results = run_eval(lm, hf_model, tasks, args.limit, args.batch_size)
                acc = extract_metric(eval_results, primary_task, metric)
                if acc is None:
                    raise RuntimeError(
                        f"metric {metric!r} missing for layer={layer_id} r={ratio}: "
                        f"{eval_results.get('results', {})}"
                    )
                print(f"[step1] layer={layer_id} r={ratio:.1f} -> Acc={acc:.6f}", flush=True)

            ratio_key = _ratio_key(ratio)
            layer_entry["acc_by_ratio"][ratio_key] = acc
            if acc_full is not None:
                layer_entry["drop_by_ratio"][ratio_key] = acc_full - acc
            store["layers"][layer_key] = layer_entry
            save_results(results_path, store)

        clear_post_rope_topk_layers(hf_model)

    finalize_step1_outputs(
        store,
        output_dir,
        layer_ids=layer_ids,
        ratios=ratios,
        acc_full=acc_full,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
