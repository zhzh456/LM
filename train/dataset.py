"""TOMATO dataset for Qwen3-VL SFT (aligned with lmms_eval tomato task)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Reuse eval prompt construction
from lmms_eval.tasks.tomato.utils import NUM_FRAMES, construct_prompt

DEFAULT_DATASET = "lmms-lab/TOMATO"
DEFAULT_CACHE_SUBDIR = "TOMATO"
DEFAULT_SYSTEM_PROMPT = "You are an expert in understanding dynamics of objects."
DEFAULT_MAX_PIXELS = 131072
QWEN_DEFAULT_MIN_PIXELS = 256 * 28 * 28
QWEN_FLOOR_MIN_PIXELS = 4 * 28 * 28


def resolve_min_pixels(max_pixels: int, min_pixels: int | None = None) -> int:
    """Ensure min_pixels <= max_pixels (required by qwen_vl_utils.smart_resize)."""
    if min_pixels is None:
        if max_pixels >= QWEN_DEFAULT_MIN_PIXELS:
            min_pixels = QWEN_DEFAULT_MIN_PIXELS
        else:
            min_pixels = QWEN_FLOOR_MIN_PIXELS
    return max(QWEN_FLOOR_MIN_PIXELS, min(min_pixels, max_pixels))


def _video_cache_dir() -> Path:
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return Path(hf_home) / DEFAULT_CACHE_SUBDIR


def resolve_video_path(doc: Dict[str, Any], cache_dir: Optional[Path] = None) -> str:
    cache = cache_dir or _video_cache_dir()
    path = cache / doc["video_path"]
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}. Download TOMATO first (see README).")
    return str(path)


def build_messages(
    doc: Dict[str, Any],
    video_path: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    num_frames: int = NUM_FRAMES,
    min_pixels: int | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> List[Dict[str, Any]]:
    min_pixels = resolve_min_pixels(max_pixels, min_pixels)
    prompt, _, _ = construct_prompt(
        question=doc["question"],
        options=doc["options"],
        num_frames=num_frames,
    )
    answer_letter = chr(65 + int(doc["answer"]))
    video_kwargs = {
        "min_pixels": min_pixels,
        "nframes": num_frames,
        "max_pixels": max_pixels,
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, **video_kwargs},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": answer_letter},
    ]


def load_tomato_split(
    split: str = "test",
    *,
    dataset_name: str = DEFAULT_DATASET,
    limit: Optional[int] = None,
    train_ratio: Optional[float] = None,
    seed: int = 42,
):
    """Load TOMATO from Hugging Face. Only ``test`` exists; use train_ratio to hold out eval."""
    ds = load_dataset(dataset_name, split=split, token=True)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    if train_ratio is None:
        return ds, None
    ds = ds.shuffle(seed=seed)
    n_train = int(len(ds) * train_ratio)
    return ds.select(range(n_train)), ds.select(range(n_train, len(ds)))


class TomatoSFTDataset:
    def __init__(
        self,
        hf_dataset,
        *,
        cache_dir: Optional[Path] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        num_frames: int = NUM_FRAMES,
        min_pixels: int | None = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ):
        self.data = hf_dataset
        self.cache_dir = cache_dir
        self.system_prompt = system_prompt
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.min_pixels = resolve_min_pixels(max_pixels, min_pixels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.data[idx]
        video_path = resolve_video_path(doc, self.cache_dir)
        messages = build_messages(
            doc,
            video_path,
            system_prompt=self.system_prompt,
            num_frames=self.num_frames,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return {"messages": messages, "id": doc.get("id", idx)}
