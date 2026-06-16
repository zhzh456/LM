#!/usr/bin/env python3
"""
First decode-step calibration: pre-RoPE vs post-RoPE top-p key relative distances.

Prefill the prompt, then run one decode forward (first generated position).
Statistics are for the first decode token's attention over all prior keys (and itself):
top-p set2 \\ set1 extra relative distances q_abs - k.
"""

from __future__ import annotations

import argparse
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

from collator import _vision_encode
from dataset import (
    DEFAULT_MAX_PIXELS,
    TomatoSFTDataset,
    load_tomato_split,
    resolve_min_pixels,
)
from sparse_attention import qk_pre_softmax_scores

from top_p_distance_calibrate import (
    ExtraDistanceAccumulator,
    _get_language_model,
    _head_distance_counts,
    _iter_text_attention_modules,
    _nucleus_mask_bhqk,
    _set_calib_pad_mask,
    process_capture,
    save_calibration_results,
)

QUERY_SCOPE = "decode"


def _build_decode_valid_mask(
    batch_size: int,
    kv_len: int,
    q_abs: torch.Tensor,
    pad_mask_2d: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Bool (B, 1, 1, K): causal decode keys + prompt padding mask."""
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


def _accumulate_decode_first_extras(
    score_pre: torch.Tensor,
    score_post: torch.Tensor,
    *,
    top_p: float,
    pad_mask_2d: torch.Tensor | None,
) -> dict[int, dict[int, int]]:
    """Decode first step: scores (B, H, 1, K), single query row index 0."""
    if score_pre.size(-2) != 1:
        raise ValueError(f"expected q_len==1 for decode capture, got {score_pre.size(-2)}")

    batch_size = score_pre.size(0)
    n_heads = score_pre.size(1)
    kv_len = score_pre.size(-1)
    device = score_pre.device
    q_abs = torch.full((batch_size,), kv_len - 1, dtype=torch.long, device=device)

    rows_pre = score_pre[:, :, 0, :]
    rows_post = score_post[:, :, 0, :]
    valid_bhqk = _build_decode_valid_mask(batch_size, kv_len, q_abs, pad_mask_2d, device)

    nuc_pre = _nucleus_mask_bhqk(rows_pre.unsqueeze(2), top_p, valid_bhqk).squeeze(2)
    nuc_post = _nucleus_mask_bhqk(rows_post.unsqueeze(2), top_p, valid_bhqk).squeeze(2)

    k_idx = torch.arange(kv_len, device=device).view(1, 1, kv_len)
    rel_dist = (q_abs.view(batch_size, 1, 1) - k_idx).expand(batch_size, n_heads, kv_len)
    valid_bh = valid_bhqk.squeeze(2)
    extra = nuc_post & ~nuc_pre & valid_bh
    return _head_distance_counts(extra.unsqueeze(2), rel_dist.unsqueeze(2))


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
    """Teacher-force the first answer token (prefill + one decode step)."""
    prompt_inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in prompt_batch.items()
    }
    pad_mask_2d = prompt_inputs.get("attention_mask")
    if pad_mask_2d is not None and pad_mask_2d.dim() == 2:
        _set_calib_pad_mask(model, pad_mask_2d)
    else:
        _set_calib_pad_mask(model, None)

    prefill_out = model(**prompt_inputs, use_cache=True, return_dict=True)
    answer_id = _first_answer_token_id(processor, messages)
    decode_ids = torch.tensor([[answer_id]], dtype=torch.long, device=device)
    decode_attn = torch.cat(
        [prompt_inputs["attention_mask"], torch.ones_like(decode_ids, dtype=prompt_inputs["attention_mask"].dtype)],
        dim=-1,
    )
    if decode_attn.dim() == 2:
        _set_calib_pad_mask(model, decode_attn)
    model(
        input_ids=decode_ids,
        attention_mask=decode_attn,
        past_key_values=prefill_out.past_key_values,
        use_cache=True,
        return_dict=True,
    )


def _reset_decode_calib_state(model: torch.nn.Module) -> None:
    model._calib_decode_captured = False
    for attn in _iter_text_attention_modules(model):
        if getattr(attn, "_calib_decode_capture_patched", False):
            attn._calib_pre_rope_key_cache = None
            attn._calib_post_rope_key_cache = None


def install_decode_score_capture_hooks(
    model: torch.nn.Module,
    capture: dict,
    *,
    layer_id: int = 0,
    top_p: float = 0.95,
) -> None:
    """Patch one layer: cache pre-RoPE K on prefill; capture first decode step only."""

    for layer_idx, attn in enumerate(_iter_text_attention_modules(model)):
        if layer_idx != layer_id:
            continue
        if getattr(attn, "_calib_decode_capture_patched", False):
            continue

        orig_forward = attn.forward

        def make_forward(_layer_idx: int, _orig_forward, _top_p: float, _model: torch.nn.Module):
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
                    self._calib_pre_rope_key_cache = key_pre_step.detach()
                    self._calib_post_rope_key_cache = key_post_step.detach()
                    key_pre = key_pre_step
                    key_post = key_post_step
                else:
                    cached_pre = getattr(self, "_calib_pre_rope_key_cache", None)
                    cached_post = getattr(self, "_calib_post_rope_key_cache", None)
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
                    and not getattr(_model, "_calib_decode_captured", False)
                ):
                    score_pre = qk_pre_softmax_scores(
                        self, query_pre, key_pre, attention_mask, self.scaling
                    )
                    score_post = qk_pre_softmax_scores(
                        self, query_states, key_post, attention_mask, self.scaling
                    )
                    pad_mask_2d = getattr(self, "_calib_pad_mask_2d", None)
                    head_counts = _accumulate_decode_first_extras(
                        score_pre,
                        score_post,
                        top_p=_top_p,
                        pad_mask_2d=pad_mask_2d,
                    )
                    del score_pre, score_post
                    capture[_layer_idx] = {
                        "head_counts": head_counts,
                        "batch_size": int(query_pre.size(0)),
                        "kv_len": kv_len,
                    }
                    _model._calib_decode_captured = True

                return _orig_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )

            return forward

        attn.forward = make_forward(layer_idx, orig_forward, top_p, model).__get__(attn, type(attn))
        attn._calib_decode_capture_patched = True


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
        description="First decode-token top-p distance calibration (pre-RoPE vs post-RoPE QK)"
    )
    p.add_argument("--model_path", type=str, default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--dataset", type=str, default="lmms-lab/TOMATO")
    p.add_argument("--output_dir", type=str, default="/tmp/qwen3vl-calibration-top-p")
    p.add_argument("--dataset_fraction", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--layer_id", type=int, default=0)
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_interval", type=int, default=0)
    p.add_argument("--max_distance_plot", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.min_pixels = resolve_min_pixels(args.max_pixels, args.min_pixels)

    out_dir = Path(args.output_dir) / QUERY_SCOPE
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
    install_decode_score_capture_hooks(
        model,
        capture,
        layer_id=args.layer_id,
        top_p=args.top_p,
    )

    dataset = TomatoSFTDataset(
        train_hf,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    collator = Qwen3VLDecodePromptCollator(processor=processor)
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
        "query_scope": QUERY_SCOPE,
        "teacher_forced_first_answer_token": True,
    }

    batch_size = max(1, int(args.batch_size))
    model.eval()
    print(
        f"[calib-decode] samples={len(dataset)} batch_size={batch_size} top_p={args.top_p} "
        f"layer_id={args.layer_id} (teacher-forced 1st answer token) "
        f"heads={n_heads} output={out_dir}",
        flush=True,
    )

    save_interval = max(0, int(args.save_interval))
    last_saved_at = 0

    for sample_idx in tqdm(range(len(dataset)), desc="calibrate-decode"):
        capture.clear()
        _reset_decode_calib_state(model)

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
            print(
                f"[calib-decode] warn: no capture on layer {args.layer_id} sample {sample_idx}, skip",
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
    print(f"[calib-decode] final -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
