"""Text-only dataset utilities for budget training on LLM data."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset


def _load_one_dataset(
    dataset_name: str,
    *,
    split: str,
    text_field: str,
    limit: Optional[int],
) -> Dataset:
    if os.path.isdir(dataset_name):
        data_dir = Path(dataset_name)
        jsonl_files = sorted(str(p) for p in data_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in local dataset dir: {dataset_name}")
        if limit is not None:
            # For huge local jsonl corpora, stream and materialize only `limit` rows
            # to avoid writing a full Arrow cache shard.
            it = load_dataset("json", data_files={"train": jsonl_files}, split="train", streaming=True, token=None)
            rows = []
            for i, row in enumerate(it):
                rows.append(row)
                if i + 1 >= limit:
                    break
            ds = Dataset.from_list(rows)
        else:
            ds = load_dataset("json", data_files={"train": jsonl_files}, split="train", token=None)
    else:
        hf_token = os.getenv("HF_TOKEN", None)
        ds = load_dataset(dataset_name, split=split, token=hf_token)
    if limit is not None and not os.path.isdir(dataset_name):
        ds = ds.select(range(min(limit, len(ds))))
    if text_field not in ds.column_names:
        raise ValueError(
            f"text_field='{text_field}' not found in dataset='{dataset_name}'. "
            f"Available fields: {ds.column_names}"
        )
    if text_field != "text":
        ds = ds.rename_column(text_field, "text")
    keep_cols = ["text"] + (["id"] if "id" in ds.column_names else [])
    return ds.select_columns(keep_cols)


@dataclass
class TextDataSpec:
    dataset: str
    split: str = "train"
    text_field: str = "text"
    limit: Optional[int] = None


def load_text_mix_split(
    base_spec: TextDataSpec,
    *,
    long_spec: Optional[TextDataSpec] = None,
    train_ratio: float = 1.0,
    seed: int = 42,
):
    """
    Load one or two text datasets and optionally split into train/eval.

    If ``long_spec`` is provided, base + long datasets are concatenated.
    """
    base_ds = _load_one_dataset(
        base_spec.dataset,
        split=base_spec.split,
        text_field=base_spec.text_field,
        limit=base_spec.limit,
    )
    if long_spec is not None:
        long_ds = _load_one_dataset(
            long_spec.dataset,
            split=long_spec.split,
            text_field=long_spec.text_field,
            limit=long_spec.limit,
        )
        ds = concatenate_datasets([base_ds, long_ds])
    else:
        ds = base_ds

    if train_ratio >= 1.0:
        return ds, None

    ds = ds.shuffle(seed=seed)
    n_train = min(int(len(ds) * train_ratio), len(ds))
    train_ds = ds.select(range(n_train))
    eval_ds = ds.select(range(n_train, len(ds))) if n_train < len(ds) else None
    return train_ds, eval_ds


class TextSFTDataset:
    """Return raw text records for a collator-driven CausalLM objective."""

    def __init__(self, hf_dataset, *, text_field: str = "text"):
        self.data = hf_dataset
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        text = item.get(self.text_field, "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        return {"text": text, "id": item.get("id", idx)}
