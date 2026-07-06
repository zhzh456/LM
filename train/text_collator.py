"""Text-only batch collator for CausalLM loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoTokenizer


@dataclass
class TextCausalCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [f.get("text", "") for f in features]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
