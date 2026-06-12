"""Probe layer-0 attention Q/K and distill tensor memory (one sample, no backward)."""

from __future__ import annotations

import argparse
import gc

import torch
from dataset import TomatoSFTDataset, load_tomato_split
from patch_sparse_attn import patch_model_for_sparse_training, set_run_distill_this_step
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from train import Qwen3VLDataCollator


def _fmt_gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def _mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    print(f"[mem] {tag}: allocated={_fmt_gb(alloc)} reserved={_fmt_gb(reserv)}", flush=True)


def _tensor_gb(shape: tuple[int, ...], dtype: torch.dtype) -> float:
    n = 1
    for d in shape:
        n *= d
    bpe = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    return n * bpe / 1024**3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--max_pixels", type=int, default=12845056)
    p.add_argument("--min_pixels", type=int, default=200704)
    p.add_argument("--num_frames", type=int, default=16)
    args = p.parse_args()

    captured: dict = {}

    def hook(attn, hidden_states, *a, **kw):
        q_len = hidden_states.shape[1]
        n_heads = attn.config.num_attention_heads
        kv_len = q_len
        captured["q_len"] = q_len
        captured["kv_len"] = kv_len
        captured["n_heads"] = n_heads
        shape = (1, n_heads, q_len, kv_len)
        print(
            f"[probe] layer0 hidden seq={q_len} heads={n_heads} " f"attn map {shape} bf16={_tensor_gb(shape, torch.bfloat16):.2f}GiB " f"fp32={_tensor_gb(shape, torch.float32):.2f}GiB",
            flush=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    patch_model_for_sparse_training(model, layer_id=0, sparse_kl_weight=0.1, sparse_dist_score_scale=0.1)

    from patch_sparse_attn import _iter_text_attention_modules

    for i, attn in enumerate(_iter_text_attention_modules(model)):
        if i == 0:
            orig = attn.forward

            def wrapped(self, hidden_states, *a, **kw):
                hook(self, hidden_states)
                return orig(hidden_states, *a, **kw)

            attn.forward = wrapped.__get__(attn, type(attn))
            break

    train_hf, _ = load_tomato_split(limit=1)
    ds = TomatoSFTDataset(train_hf, num_frames=args.num_frames, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    collator = Qwen3VLDataCollator(processor=processor)
    batch = collator([ds[0]])

    model = model.cuda().train()
    set_run_distill_this_step(model, True)
    _mem("after model load")
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after empty_cache")

    with torch.no_grad():
        model(**{k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items() if k != "labels"})

    _mem("after forward (no grad)")


if __name__ == "__main__":
    main()
