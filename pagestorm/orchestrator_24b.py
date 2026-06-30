from __future__ import annotations

from typing import Mapping

from .artifacts_24b import BOOK_PLAN_PREVIEW_REGEX_24B, BOOK_PLAN_REMAINDER_REGEX_24B
from .models import GenerationRun, StageOutput, StageOutputCallback, StageStartCallback, StageTextChunkCallback, StoryBundle
from .orchestrator import (
    BOOK_PREVIEW_TEMPERATURE,
    DEFAULT_STAGE_TEMPERATURE,
    _StageGenerationResult,
    _append_stage_roles,
    _build_generation_run,
    _count_appended_text_tokens,
    _ensure_trailing_newline,
    _first_chapter_stage_order,
    _generate_rendered_prompt_text_with_metrics,
    _has_non_default_removed_stage_override,
)
from .vllm_runtime import StoryVllmRuntime, close_generation_runtime


def generate_book_plan_text_with_metrics(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_max_tokens: Mapping[str, int],
    use_guidance: bool,
    seed: int | None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> _StageGenerationResult:
    rendered_prompt = bundle.protocol.render_stage_prompt(prompt, completed_stages, "book_plan")
    remaining_stage_tokens = stage_max_tokens["book_plan"]

    preview_generation_result = _generate_rendered_prompt_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        rendered_prompt=rendered_prompt,
        stage_role="book_plan",
        stage_max_tokens={"book_plan": remaining_stage_tokens},
        structured_outputs=None if not use_guidance else {"regex": BOOK_PLAN_PREVIEW_REGEX_24B},
        seed=seed,
        temperature_override=BOOK_PREVIEW_TEMPERATURE,
        top_k_override=20,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )
    preview_text = _ensure_trailing_newline(preview_generation_result.text)
    if stage_text_chunk_callback is not None and preview_text != preview_generation_result.text:
        stage_text_chunk_callback(preview_text[len(preview_generation_result.text) :])
    remaining_stage_tokens -= _count_appended_text_tokens(bundle, rendered_prompt, preview_text)
    if remaining_stage_tokens <= 0:
        raise ValueError("24b book_plan exhausted the stage token budget before the world-rules section completed.")

    remainder_generation_result = _generate_rendered_prompt_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        rendered_prompt=rendered_prompt + preview_text,
        stage_role="book_plan",
        stage_max_tokens={"book_plan": remaining_stage_tokens},
        structured_outputs=None if not use_guidance else {"regex": BOOK_PLAN_REMAINDER_REGEX_24B},
        seed=None if seed is None else seed + 1,
        temperature_override=DEFAULT_STAGE_TEMPERATURE,
        top_k_override=None,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )
    return _StageGenerationResult(
        text=preview_text + remainder_generation_result.text,
        prompt_token_count=remainder_generation_result.prompt_token_count,
        max_tokens_requested=preview_generation_result.max_tokens_requested + remainder_generation_result.max_tokens_requested,
    )


def generate_first_chapter(
    bundle: StoryBundle,
    *,
    prompt: str,
    stage_max_tokens: Mapping[str, int] | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    stage_output_callback: StageOutputCallback | None = None,
    stage_start_callback: StageStartCallback | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> GenerationRun:
    from . import orchestrator

    if _has_non_default_removed_stage_override(bundle, stage_max_tokens, "book_preview"):
        raise ValueError("24b does not expose a standalone book_preview stage. Use book_plan token overrides instead.")
    resolved_stage_max_tokens = bundle.profile.resolve_stage_max_tokens(stage_max_tokens)
    runtime = orchestrator.create_generation_runtime(
        bundle,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    stage_outputs: list[StageOutput] = []
    completed_stages: list[tuple[str, str]] = []
    try:
        failed_stage_role = _append_stage_roles(
            bundle=bundle,
            runtime=runtime,
            stage_outputs=stage_outputs,
            completed_stages=completed_stages,
            stage_roles=_first_chapter_stage_order(bundle),
            prompt=prompt,
            stage_max_tokens=resolved_stage_max_tokens,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        return _build_generation_run(
            bundle=bundle,
            prompt=prompt,
            model_identifier=runtime.model_identifier,
            stage_outputs=stage_outputs,
            later_chapter_count=0,
            failed_stage_role=failed_stage_role,
        )
    finally:
        close_generation_runtime(runtime)
