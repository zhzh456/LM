#!/usr/bin/env python3
"""Train per-head relative-position bias for sparse attention on Qwen3-VL (TOMATO)."""

from __future__ import annotations

import argparse
import os

import torch
from collator import Qwen3VLDataCollator
from dataset import (
    DEFAULT_MAX_PIXELS,
    TomatoSFTDataset,
    load_tomato_split,
    resolve_min_pixels,
)
from patch_sparse_attn import (
    SPARSE_REL_POS_FILENAME,
    _get_language_model,
    collect_sparse_distill_losses,
    finalize_sparse_distill_losses,
    iter_attn_with_bias,
    load_sparse_rel_pos_checkpoint,
    patch_model_for_sparse_training,
    save_sparse_rel_pos_checkpoint,
    set_run_distill_this_step,
    trainable_sparse_parameters,
)
from transformers import (
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train sparse-attention rel-pos bias (16K/head), backbone frozen")
    p.add_argument(
        "--model_path",
        type=str,
        default="/home/zhanghao360/model/Qwen3-VL-4B-Instruct",
    )
    p.add_argument("--output_dir", type=str, default="./train/outputs/qwen3vl-sparse-attn")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume Trainer state (model/optimizer/scheduler) from checkpoint dir.",
    )
    p.add_argument("--dataset", type=str, default="lmms-lab/TOMATO")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Fraction of data for training; 1.0 = all data, no eval. Use e.g. 0.9 to hold out 10%% for Trainer eval.",
    )
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument(
        "--sparse_topk_k",
        type=int,
        default=1000,
        help="Distance-branch top-k per query (union with content top-k).",
    )
    p.add_argument(
        "--content_topk_k",
        type=int,
        default=500,
        help="Content-branch top-k per query (pre-RoPE Q·K, hard detach).",
    )
    p.add_argument("--ste_tau", type=float, default=0.25, help="STE softmax temperature for distance top-k.")
    p.add_argument("--sparse_gap_recall_weight", type=float, default=1.0)
    p.add_argument(
        "--sparse_dist_score_scale",
        type=float,
        default=0.75,
        help="Add scale*mask_d*S[d] to RoPE QK logits (grad path for S[d]).",
    )
    p.add_argument(
        "--sparse_topk_ratio",
        type=float,
        default=0.2,
        help="Deprecated; kept for checkpoint compatibility only.",
    )
    p.add_argument("--rel_pos_buckets", type=int, default=16384, help="Per-head bias length (16K)")
    p.add_argument("--rel_pos_near_tau", type=float, default=128.0, help="Init decay for distance (smaller=sharper near bias)")
    p.add_argument("--rel_pos_wave_period", type=float, default=32.0, help="Init oscillation period in tokens")
    p.add_argument("--rel_pos_wave_amp", type=float, default=0.12, help="Init oscillation amplitude")
    p.add_argument("--rel_pos_noise_std", type=float, default=0.01, help="Per-head init noise")
    p.add_argument(
        "--train_layer_id",
        type=int,
        default=35,
        help="Text decoder layer index to train sparse rel-pos on (0-based)",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="HF attention for non-sparse layers (sparse layer uses custom forward)",
    )
    p.add_argument(
        "--distill_every_n_steps",
        type=int,
        default=1,
        help="Run dense teacher every N training steps",
    )
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument(
        "--save_every_epoch_fraction",
        type=float,
        default=None,
        help="If set (e.g. 0.5), save every fraction of an epoch; overrides save_strategy/save_steps.",
    )
    p.add_argument("--save_at_end", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--logging_steps", type=int, default=1, help="Log every N optimizer steps")
    p.add_argument("--logging_first_step", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument(
        "--baseline_plain_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use plain model attention (no rel-pos patch), no parameter updates; report baseline loss only.",
    )
    p.add_argument(
        "--rel_pos_init_path",
        type=str,
        default=None,
        help="Optional .pt with keys layer_{i}.head_{h}; load into rel-pos bias after patch (no collection).",
    )
    return p.parse_args()


def _input_length_stats(inputs: dict) -> dict[str, int]:
    return {"seq_len": int(inputs["input_ids"].shape[1])}


def _format_step_log(metrics: dict) -> str:
    loss = metrics.get("loss")
    lr = metrics.get("learning_rate")
    grad = metrics.get("grad_norm")
    epoch = metrics.get("epoch")
    if lr is None:
        lr = 0.0
    if grad is None:
        grad = 0.0
    if epoch is None:
        epoch = 0.0
    return f"[train] seq_len={metrics.get('seq_len', 'na')} " f"loss={loss:.6f} " f"learning_rate={lr:.6g} " f"epoch={epoch:.4f} " f"grad_norm={grad:.6f}"


class ConsoleLossCallback(TrainerCallback):
    """Pretty-print epoch/step/loss to stdout every optimizer step."""

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(
                f"[train] start | max_steps={state.max_steps} epochs={args.num_train_epochs} " f"logging_steps={args.logging_steps} grad_accum={args.gradient_accumulation_steps}",
                flush=True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass  # Trainer printing is handled by SparseAttentionTrainer.on_log


class SavePathCallback(TrainerCallback):
    """On each Trainer checkpoint: save sparse_rel_pos_bias.pt and print path."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_sparse_rel_pos_checkpoint(self.model, ckpt_dir)


class SparseAttentionTrainer(Trainer):
    def __init__(
        self,
        *args,
        distill_every_n_steps: int = 1,
        baseline_plain_attention: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distill_every_n_steps = max(1, distill_every_n_steps)
        self.baseline_plain_attention = baseline_plain_attention
        self._micro_in_epoch = 0
        self._last_step_metrics: dict[str, float | int | bool] = {}
        # Silence default dict-style logging callbacks; keep only custom one-line print.
        try:
            from transformers.trainer_callback import PrinterCallback

            self.remove_callback(PrinterCallback)
        except Exception:
            pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._micro_in_epoch = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._micro_in_epoch += 1
        grad_accum = max(1, self.args.gradient_accumulation_steps)
        micro_idx = ((self._micro_in_epoch - 1) % grad_accum) + 1
        optimizer_step = self.state.global_step + (1 if self.accelerator.sync_gradients else 0)

        training = model.training
        length_stats = _input_length_stats(inputs)
        self._last_seq_len = length_stats["seq_len"]

        if self.baseline_plain_attention:
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Baseline eval needs labels for CE loss.")
            log_dict = {"loss": float(loss.detach().item())}
            self._last_step_metrics = {"loss_total": log_dict["loss"], "seq_len": length_stats["seq_len"]}
            return (loss, outputs) if return_outputs else loss

        run_distill = (not training) or (self.state.global_step % self.distill_every_n_steps == 0)
        set_run_distill_this_step(model, run_distill)

        forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**forward_inputs)
        finalize_sparse_distill_losses(model)

        parts = collect_sparse_distill_losses(model)
        distill_raw = parts["distill"]
        if distill_raw is None or not run_distill:
            raise RuntimeError("No distill loss on this step; set distill_every_n_steps=1 or enable teacher forward.")
        loss = distill_raw
        log_dict: dict[str, float] = {
            "loss": float(loss.detach().item()),
        }
        if parts.get("gap") is not None:
            log_dict["loss_gap"] = float(parts["gap"].detach().item())
        if parts.get("gap_recall") is not None:
            log_dict["gap_recall"] = float(parts["gap_recall"].detach().item())
        if parts.get("union_recall") is not None:
            log_dict["union_recall"] = float(parts["union_recall"].detach().item())
        log_dict["distill_active"] = int(run_distill)
        self._last_step_metrics = {
            "loss_total": log_dict["loss"],
            "loss_gap": log_dict.get("loss_gap"),
            "gap_recall": log_dict.get("gap_recall"),
            "union_recall": log_dict.get("union_recall"),
            "distill_active": run_distill,
            "seq_len": length_stats["seq_len"],
        }

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        import gc

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if not self.baseline_plain_attention:
            for attn in iter_attn_with_bias(model):
                attn._sparse_kl_loss = None
                attn._sparse_mse_loss = None
                attn._sparse_gap_recall_loss = None
                attn._sparse_gap_recall = None
                attn._sparse_topk_recall = None
                attn._sparse_distill_loss = None
                attn._sparse_distill_extras = None
                attn._sparse_distill_attention_mask = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """Print total loss and each component every logging step."""
        if not logs:
            return
        epoch = float(logs.get("epoch", self.state.epoch or 0.0))
        step = int(logs.get("step", self.state.global_step))
        m = self._last_step_metrics
        loss_total = float(m.get("loss_total", logs.get("loss", float("nan"))))
        lr = logs.get("learning_rate")
        if lr is None and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        grad = logs.get("grad_norm", 0.0)

        def _fmt(v: float) -> str:
            return "nan" if v != v else f"{v:.6f}"

        loss_gap = m.get("loss_gap")
        gap_recall = m.get("gap_recall")
        union_recall = m.get("union_recall")
        distill_on = bool(m.get("distill_active", False))
        seq_len = m.get("seq_len", "na")

        extra = ""
        if loss_gap is not None:
            extra += f" loss_gap={_fmt(float(loss_gap))}"
        if gap_recall is not None:
            extra += f" gap_recall={_fmt(float(gap_recall))}"
        if union_recall is not None:
            extra += f" union_recall={_fmt(float(union_recall))}"

        print(
            f"[train] step={step} epoch={epoch:.4f} seq_len={seq_len} "
            f"loss={_fmt(loss_total)}{extra} distill={int(distill_on)} "
            f"lr={float(lr):.6g} grad_norm={float(grad):.6f}",
            flush=True,
        )

        logs = dict(logs)
        logs["epoch"] = epoch
        logs["step"] = step
        logs.update({k: v for k, v in m.items() if isinstance(v, (int, float)) and k != "distill_active"})
        logs["distill_active"] = int(distill_on)
        self.state.log_history.append(logs)


def _enable_cuda_speedups() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    _enable_cuda_speedups()
    args.min_pixels = resolve_min_pixels(args.max_pixels, args.min_pixels)
    mode = "baseline_plain_attention" if args.baseline_plain_attention else "position_only_training"
    print(f"Run mode={mode} | max_pixels={args.max_pixels} min_pixels={args.min_pixels} " f"train_layer_id={args.train_layer_id} attn={args.attn_implementation} " f"distill_every_n_steps={args.distill_every_n_steps}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_hf, eval_hf = load_tomato_split(
        split="test",
        dataset_name=args.dataset,
        limit=args.limit,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    from transformers import AutoProcessor

    # Single GPU: pin to cuda:0 (clearer than "auto" on one card).
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
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

    common_ds_kw = dict(
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    train_dataset = TomatoSFTDataset(train_hf, **common_ds_kw)
    eval_dataset = TomatoSFTDataset(eval_hf, **common_ds_kw) if eval_hf is not None and len(eval_hf) > 0 else None

    if not args.baseline_plain_attention:
        patch_model_for_sparse_training(
            model,
            layer_id=args.train_layer_id,
            num_buckets=args.rel_pos_buckets,
            near_tau=args.rel_pos_near_tau,
            wave_period=args.rel_pos_wave_period,
            wave_amp=args.rel_pos_wave_amp,
            noise_std=args.rel_pos_noise_std,
            sparse_topk_k=args.sparse_topk_k,
            content_topk_k=args.content_topk_k,
            target_topk_k=args.sparse_topk_k,
            ste_tau=args.ste_tau,
            sparse_gap_recall_weight=args.sparse_gap_recall_weight,
            sparse_dist_score_scale=args.sparse_dist_score_scale,
        )

        trainable = trainable_sparse_parameters(model)
        n_params = sum(p.numel() for p in trainable)
        n_layers = len(list(iter_attn_with_bias(model)))
        heads_per_layer = len(trainable) // max(n_layers, 1)
        n_text_layers = len(_get_language_model(model).layers)
        print(
            f"Trainable: layer {args.train_layer_id} x {heads_per_layer} heads " f"= {len(trainable)} vectors x {args.rel_pos_buckets} dims, {n_params:,} total",
            flush=True,
        )
        if args.train_layer_id + 1 < n_text_layers:
            print(
                f"[sparse] early-stop text decoder at layer {args.train_layer_id} " f"(skip layers {args.train_layer_id + 1}-{n_text_layers - 1}, no lm_head)",
                flush=True,
            )
        if args.rel_pos_init_path:
            n_loaded = load_sparse_rel_pos_checkpoint(
                model,
                args.rel_pos_init_path,
                expected_layer_id=args.train_layer_id,
            )
            print(
                f"[init] loaded {n_loaded} head vectors from {args.rel_pos_init_path} " f"(train_layer_id={args.train_layer_id})",
                flush=True,
            )
        else:
            print(
                f"[init] default near_tau={args.rel_pos_near_tau} " f"wave_period={args.rel_pos_wave_period} wave_amp={args.rel_pos_wave_amp}",
                flush=True,
            )

    steps_per_epoch = max(
        1,
        (len(train_dataset) + args.per_device_train_batch_size * args.gradient_accumulation_steps - 1) // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
    )
    print(
        f"Dataset: train={len(train_dataset)} eval={len(eval_dataset) if eval_dataset else 0} | " f"~{steps_per_epoch} optimizer step(s)/epoch " f"(batch={args.per_device_train_batch_size} grad_accum={args.gradient_accumulation_steps})",
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
        logging_first_step=args.logging_first_step,
        logging_strategy="steps",
        log_level="warning",
        log_level_replica="warning",
        disable_tqdm=False,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=args.report_to,
        seed=args.seed,
        label_names=["labels"],
    )

    trainer = SparseAttentionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(processor=processor),
        callbacks=[ConsoleLossCallback(), SavePathCallback(model)],
        distill_every_n_steps=args.distill_every_n_steps,
        baseline_plain_attention=args.baseline_plain_attention,
    )

    if args.baseline_plain_attention:
        # Baseline: no parameter updates, just compute plain-attention loss on dataset.
        eval_for_baseline = train_dataset if len(train_dataset) > 0 else eval_dataset
        metrics = trainer.evaluate(eval_dataset=eval_for_baseline)
        print(
            f"[baseline] loss={metrics.get('eval_loss', float('nan')):.6f} " f"samples={len(eval_for_baseline) if eval_for_baseline is not None else 0}",
            flush=True,
        )
        return

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None:
        resume_ckpt = str(resume_ckpt).strip()
        if resume_ckpt.lower() in ("", "none", "null", "false"):
            resume_ckpt = None
        elif not os.path.isdir(resume_ckpt):
            raise FileNotFoundError(f"resume_from_checkpoint not found: {resume_ckpt}")
    if resume_ckpt:
        print(f"[train] resume_from_checkpoint={resume_ckpt}", flush=True)
        if not args.baseline_plain_attention:
            ckpt_pt = os.path.join(resume_ckpt, SPARSE_REL_POS_FILENAME)
            if os.path.isfile(ckpt_pt):
                n_loaded = load_sparse_rel_pos_checkpoint(
                    model,
                    ckpt_pt,
                    expected_layer_id=args.train_layer_id,
                )
                print(
                    f"[train] restored {n_loaded} head vectors from {ckpt_pt}",
                    flush=True,
                )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    if args.save_at_end:
        final_dir = os.path.join(args.output_dir, "final")
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        save_sparse_rel_pos_checkpoint(model, final_dir)
        print(f"[train] saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
