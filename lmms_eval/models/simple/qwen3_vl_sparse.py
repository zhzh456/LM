"""Qwen3-VL with trained per-head relative-position attention for lmms_eval."""

from __future__ import annotations

from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
from lmms_eval.models.sparse_qwen3_setup import setup_sparse_attention_for_eval


@register_model("qwen3_vl_sparse")
class Qwen3_VL_Sparse(Qwen3_VL):
    """
    Eval: one text layer (sparse_layer_id) uses sparse decode rel-pos; others use default attention.
    Training uses train_layer_id (see train/patch_sparse_attn.py).

    Default attn_implementation=flash_attention_2 on non-sparse layers;
    sparse_layer_id uses custom rel-pos forward.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        sparse_rel_pos_path: str = "",
        rel_pos_buckets: int = 4096,
        sparse_layer_id: int = 0,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 16,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        log_input_length: bool = True,
        print_generation: bool = True,
        save_attn_scores_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not sparse_rel_pos_path:
            raise ValueError(
                "qwen3_vl_sparse requires sparse_rel_pos_path "
                "(e.g. /tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt)"
            )
        if kwargs:
            raise ValueError(f"Unexpected kwargs for qwen3_vl_sparse: {kwargs}")

        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            fps=fps,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            enable_thinking=enable_thinking,
            reasoning_prompt=reasoning_prompt,
            log_input_length=log_input_length,
            print_generation=print_generation,
            save_attn_scores_dir=None,
        )
        self.save_attn_scores_dir = save_attn_scores_dir

        setup_sparse_attention_for_eval(
            self._model,
            rel_pos_path=sparse_rel_pos_path,
            rel_pos_buckets=rel_pos_buckets,
            layer_id=sparse_layer_id,
            save_attn_scores_dir=save_attn_scores_dir,
        )
