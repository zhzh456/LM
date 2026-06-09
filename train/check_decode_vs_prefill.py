#!/usr/bin/env python3
"""Count text-attn forward steps by q_len during one generate() call."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="mmbench_en_dev", choices=["mmbench_en_dev", "tomato"])
    p.add_argument("--model_path", default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--sparse-layer-id", type=int, default=35)
    args = p.parse_args()

    from datasets import load_dataset
    from patch_sparse_attn import _iter_text_attention_modules
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    from lmms_eval.tasks import TaskManager

    tm = TaskManager()
    task_dict = tm.load_task_or_group(args.task)
    task = task_dict[args.task]
    docs = task.test_docs()
    doc = docs[0]
    text = task.doc_to_text(doc)
    visuals = task.doc_to_visual(doc)
    messages = [{"role": "user", "content": visuals + [{"type": "text", "text": text}]}]

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=12845056, min_pixels=3136, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    video_metadata_list = None
    if video_inputs is not None:
        video_inputs, video_metadata_list = map(list, zip(*video_inputs))
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata_list,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )
    enc = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in enc.items()}

    stats = {"prefill": 0, "decode": 0, "other": 0}
    sparse_q_lens: list[int] = []

    for attn in _iter_text_attention_modules(model):
        orig = attn.forward

        def hooked_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            q_len = int(query_states.size(-2))
            kv_len = int(key_states.size(-2))
            if self.layer_idx == args.sparse_layer_id:
                sparse_q_lens.append(q_len)
            if q_len > 1:
                stats["prefill"] += 1
            elif q_len == 1:
                stats["decode"] += 1
            else:
                stats["other"] += 1
            return orig(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        attn.forward = hooked_forward.__get__(attn, type(attn))

    print(f"task={args.task} input_len={enc['input_ids'].shape[1]}", flush=True)
    out = model.generate(
        **enc,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )
    new_tokens = out.shape[1] - enc["input_ids"].shape[1]
    trimmed = out[0, enc["input_ids"].shape[1] :]
    text_out = processor.decode(trimmed, skip_special_tokens=True)
    print(f"new_tokens={new_tokens} output={text_out!r}", flush=True)
    print(
        f"layer {args.sparse_layer_id} unique q_lens during generate: "
        f"{sorted(set(sparse_q_lens))}",
        flush=True,
    )
    print(f"per-layer forward counts (q>1 prefill, q==1 decode): prefill={stats['prefill']} decode={stats['decode']}", flush=True)
    n_layers = len(list(_iter_text_attention_modules(model)))
    print(f"expected prefill forwards (once): {n_layers}, decode forwards: {n_layers * max(new_tokens, 0)}", flush=True)


if __name__ == "__main__":
    main()
