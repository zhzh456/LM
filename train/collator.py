"""Batch collator for Qwen3-VL supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def _vision_encode(processor: AutoProcessor, messages: List[Dict[str, Any]], *, add_generation_prompt: bool):
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    video_metadata_list = None
    if video_inputs is not None:
        video_inputs, video_metadata_list = map(list, zip(*video_inputs))

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata_list,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )


@dataclass
class Qwen3VLDataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encodings = []
        prefix_lens = []
        for feat in features:
            messages = feat["messages"]
            enc = _vision_encode(self.processor, messages, add_generation_prompt=False)
            prefix = _vision_encode(self.processor, messages[:-1], add_generation_prompt=True)
            encodings.append(enc)
            prefix_lens.append(int(prefix["input_ids"].shape[1]))

        # Pad token-level tensors to max sequence length
        max_len = max(int(e["input_ids"].shape[1]) for e in encodings)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        input_ids = torch.full((len(encodings), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encodings), max_len), dtype=torch.long)
        mm_token_type_ids = torch.zeros((len(encodings), max_len), dtype=torch.long)
        labels = torch.full((len(encodings), max_len), -100, dtype=torch.long)

        pixel_values_videos = []
        video_grid_thw = []

        for i, (enc, prefix_len) in enumerate(zip(encodings, prefix_lens)):
            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            mm = enc.get("mm_token_type_ids", torch.zeros_like(ids))
            L = int(ids.shape[0])

            input_ids[i, :L] = ids
            attention_mask[i, :L] = attn
            mm_token_type_ids[i, :L] = mm

            labels[i, :L] = ids
            labels[i, :prefix_len] = -100
            labels[i, attention_mask[i] == 0] = -100

            if "pixel_values_videos" in enc:
                pixel_values_videos.append(enc["pixel_values_videos"])
            if "video_grid_thw" in enc:
                video_grid_thw.append(enc["video_grid_thw"])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
        }
        if pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
        if video_grid_thw:
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch
