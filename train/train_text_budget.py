#!/usr/bin/env python3
"""Train sparse budget params on text + long-context text datasets."""

from __future__ import annotations

import argparse
import os

import torch
from patch_sparse_attn import (
    _get_language_model,
    collect_sparse_distill_losses,
    iter_attn_with_bias,
    patch_model_for_sparse_training,
    save_sparse_rel_pos_checkpoint,
    trainable_sparse_parameters,
)
from text_collator import TextCausalCollator
from text_dataset import TextDataSpec, TextSFTDataset, load_text_mix_split
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


def parse_args():
    p = argparse.ArgumentParser(description="Train sparse budget on text/long-context datasets")
    p.add_argument("--model_path", type=str, default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct")
    p.add_argument("--output_dir", type=str, default="/tmp/qwen3vl-sparse-attn-text")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--dataset", type=str, required=True, help="Base text dataset on HF, e.g. allenai/c4")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_text_field", type=str, default="text")
    p.add_argument("--dataset_limit", type=int, default=None)

    p.add_argument("--long_dataset", type=str, default=None, help="Optional long-context dataset")
    p.add_argument("--long_dataset_split", type=str, default="train")
    p.add_argument("--long_dataset_text_field", type=str, default="text")
    p.add_argument("--long_dataset_limit", type=int, default=None)

    p.add_argument("--train_ratio", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_every_epoch_fraction", type=float, default=None)
    p.add_argument("--save_at_end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="none")

    p.add_argument("--train_layer_ids", type=str, default="0", help="Comma/range layer ids, e.g. '0,1' or '0-3'")
    p.add_argument("--budget_granularity", type=str, default="head", choices=["layer", "head"])
    p.add_argument("--budget_init_ratio", type=float, default=0.5)
    p.add_argument("--budget_lambda", type=float, default=0.0)
    p.add_argument("--budget_ste_temperature", type=float, default=1.0)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    return p.parse_args()


def _parse_layer_ids(spec: str, n_layers: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(p))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("train_layer_ids resolved to empty list")
    bad = [x for x in uniq if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"train_layer_ids out of range: {bad}, n_layers={n_layers}")
    return uniq


class ConsoleLossCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(
                f"[train] start | max_steps={state.max_steps} epochs={args.num_train_epochs} "
                f"logging_steps={args.logging_steps} grad_accum={args.gradient_accumulation_steps}",
                flush=True,
            )


class SavePathCallback(TrainerCallback):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_sparse_rel_pos_checkpoint(self.model, ckpt_dir)
        print(f"[save] sparse params -> {ckpt_dir}/sparse_rel_pos_bias.pt", flush=True)


class TextBudgetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_step_metrics: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if ce_loss is None:
            raise RuntimeError("Task CE mode expects labels and CE loss.")
        if ce_loss.dim() > 0:
            ce_loss = ce_loss.mean()

        parts = collect_sparse_distill_losses(model)
        budget_loss = parts.get("budget")
        budget_ratio = parts.get("ratio")
        if budget_loss is None:
            budget_loss = torch.zeros_like(ce_loss)
        loss = ce_loss + budget_loss
        self._last_step_metrics = {
            "loss_total": float(loss.detach().item()),
            "loss_task_ce": float(ce_loss.detach().item()),
            "loss_budget": float(budget_loss.detach().item()),
            "budget_ratio": float(budget_ratio.detach().item()) if budget_ratio is not None else float("nan"),
        }
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        for attn in iter_attn_with_bias(model):
            attn._sparse_kl_loss = None
            attn._sparse_mse_loss = None
            attn._sparse_distill_loss = None
            attn._sparse_budget_loss = None
            attn._sparse_budget_ratio = None
            attn._sparse_fidelity_loss = None
        return loss

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if not logs:
            return
        epoch = float(logs.get("epoch", self.state.epoch or 0.0))
        step = int(logs.get("step", self.state.global_step))
        m = self._last_step_metrics
        lr = logs.get("learning_rate")
        if lr is None and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        grad = logs.get("grad_norm", 0.0)
        print(
            f"[train] step={step} epoch={epoch:.4f} "
            f"loss={m.get('loss_total', float('nan')):.6f} "
            f"loss_task_ce={m.get('loss_task_ce', float('nan')):.6f} "
            f"loss_budget={m.get('loss_budget', float('nan')):.6f} "
            f"budget_ratio={m.get('budget_ratio', float('nan')):.6f} "
            f"lr={float(lr):.6g} grad_norm={float(grad):.6f}",
            flush=True,
        )
        self.state.log_history.append({"epoch": epoch, "step": step, **logs, **m})


def _enable_cuda_speedups() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    _enable_cuda_speedups()
    os.makedirs(args.output_dir, exist_ok=True)

    base_spec = TextDataSpec(
        dataset=args.dataset,
        split=args.dataset_split,
        text_field=args.dataset_text_field,
        limit=args.dataset_limit,
    )
    long_spec = None
    if args.long_dataset:
        long_spec = TextDataSpec(
            dataset=args.long_dataset,
            split=args.long_dataset_split,
            text_field=args.long_dataset_text_field,
            limit=args.long_dataset_limit,
        )
    train_hf, eval_hf = load_text_mix_split(
        base_spec,
        long_spec=long_spec,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    train_dataset = TextSFTDataset(train_hf)
    eval_dataset = TextSFTDataset(eval_hf) if eval_hf is not None and len(eval_hf) > 0 else None

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    # accelerate distributed launch is incompatible with device_map='auto'
    model_device_map = "auto" if world_size == 1 else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map=model_device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_text_layers = len(_get_language_model(model).layers)
    layer_ids = _parse_layer_ids(args.train_layer_ids, n_text_layers)
    print(f"[train] target_layers={','.join(map(str, layer_ids))}", flush=True)

    patch_model_for_sparse_training(
        model,
        layer_ids=layer_ids,
        training_target="budget",
        enable_early_stop=False,
        patch_causal_lm=False,
        budget_use_distill=False,
        init_ratio=args.budget_init_ratio,
        budget_lambda=args.budget_lambda,
        ste_temperature=args.budget_ste_temperature,
        budget_granularity=args.budget_granularity,
    )

    trainable = trainable_sparse_parameters(model)
    n_params = sum(p.numel() for p in trainable)
    print(
        f"Trainable: budget({args.budget_granularity}) x {len(trainable)} tensors ({n_params:,} params) "
        f"| init_ratio={args.budget_init_ratio} lambda={args.budget_lambda} ste_temp={args.budget_ste_temperature}",
        flush=True,
    )

    effective_batch = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size)
    steps_per_epoch = max(1, (len(train_dataset) + effective_batch - 1) // effective_batch)
    print(
        f"Dataset: train={len(train_dataset)} eval={len(eval_dataset) if eval_dataset else 0} | "
        f"~{steps_per_epoch} optimizer step(s)/epoch "
        f"(batch={args.per_device_train_batch_size} grad_accum={args.gradient_accumulation_steps} world_size={world_size})",
        flush=True,
    )

    save_strategy = args.save_strategy
    save_steps = args.save_steps if args.save_strategy != "no" else None
    if args.save_every_epoch_fraction is not None:
        frac = max(args.save_every_epoch_fraction, 1e-6)
        save_strategy = "steps"
        save_steps = max(1, int(round(steps_per_epoch * frac)))
        print(
            f"[save] every {frac:.4g} epoch -> save_steps={save_steps} (steps_per_epoch={steps_per_epoch})",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=args.report_to,
        seed=args.seed,
        label_names=["labels"],
    )

    trainer = TextBudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TextCausalCollator(tokenizer=tokenizer, max_length=args.max_length),
        callbacks=[ConsoleLossCallback(), SavePathCallback(model)],
    )

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt:
        print(f"[train] resume_from_checkpoint={resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if args.save_at_end:
        final_dir = os.path.join(args.output_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        save_sparse_rel_pos_checkpoint(model, final_dir)
        print(f"[train] saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
