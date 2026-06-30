from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .artifacts import render_book_text, render_story_stage_artifact
from .artifacts_14b import PlannedChapterInfo
from .models import GenerationRun, StageOutput, StageOutputCallback, StageStartCallback, StageTextChunkCallback, StoryBundle
from .vllm_runtime import StoryVllmRuntime, create_generation_runtime, generate_completion


FULL_BOOK_ONLY_STAGE_ROLES = {
    "full_book_chapters_plan",
    "book_characters_list",
    "scene_breakdown",
    "chapter_text",
}
BOOK_PREVIEW_TEMPERATURE = 0.8
CHAPTER_NAMES_TEMPERATURE = 0.8
CHAPTER_PLAN_TEMPERATURE = 0.6
COMPONENT_TEMPERATURE = 0.45
DEFAULT_STAGE_TEMPERATURE = 0.2
REQUEST_MAX_TOKEN_SAFETY_MARGIN = 256


@dataclass(frozen=True)
class _StageGenerationResult:
    text: str
    prompt_token_count: int
    max_tokens_requested: int


def _stage_temperature(stage_role: str) -> float:
    if stage_role == "book_preview":
        return BOOK_PREVIEW_TEMPERATURE
    if stage_role in {"first_chapter_plan", "full_book_chapters_plan"}:
        return CHAPTER_PLAN_TEMPERATURE
    if stage_role in {"book_plan", "book_characters_list", "scene_breakdown"}:
        return COMPONENT_TEMPERATURE
    return DEFAULT_STAGE_TEMPERATURE


def _stage_top_k(stage_role: str) -> int | None:
    if stage_role == "book_preview":
        return 20
    return None


def _escape_regex_literal(literal_text: str) -> str:
    return re.escape(literal_text).replace(r"\ ", " ")


def _python_validation_regex(regex_pattern: str) -> str:
    return regex_pattern.replace(r"\z", r"\Z")


def _first_chapter_stage_order(bundle: StoryBundle) -> tuple[str, ...]:
    return tuple(
        stage_role for stage_role in bundle.protocol.DEFAULT_STAGE_SEQUENCE if stage_role not in FULL_BOOK_ONLY_STAGE_ROLES
    )


def _uses_full_book_workflow(bundle: StoryBundle) -> bool:
    return "full_book_chapters_plan" in bundle.protocol.DEFAULT_STAGE_SEQUENCE


def _uses_preloaded_book_plan_workflow(bundle: StoryBundle) -> bool:
    return _uses_full_book_workflow(bundle) and "first_chapter_scene_breakdown" not in bundle.protocol.DEFAULT_STAGE_SEQUENCE


def _ensure_supported_full_book_workflow(bundle: StoryBundle) -> None:
    if _uses_full_book_workflow(bundle) and not _uses_preloaded_book_plan_workflow(bundle):
        raise ValueError(
            "Unsupported full-book protocol workflow. Full-book generation requires a preloaded book_plan workflow "
            "without first_chapter_scene_breakdown."
        )


def _resolve_stage_structured_outputs(
    bundle: StoryBundle,
    stage_role: str,
    structured_output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if structured_output_overrides is not None:
        overridden_structured_output = structured_output_overrides.get(stage_role)
        if overridden_structured_output is not None:
            return dict(overridden_structured_output)
    structured_outputs = bundle.protocol.stage_structured_outputs(stage_role)
    if structured_outputs is None:
        return None
    return structured_outputs


def _build_stage_output(
    bundle: StoryBundle,
    stage_role: str,
    stage_text: str,
    structured_output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> StageOutput:
    structured_output = _resolve_stage_structured_outputs(bundle, stage_role, structured_output_overrides)
    regex_name: str | None = None
    regex_valid: bool | None = None
    validation_error: str | None = None
    if structured_output is not None:
        regex_pattern = structured_output.get("regex")
        if not isinstance(regex_pattern, str):
            raise TypeError(f"story stage protocol structured_outputs for {stage_role!r} must include a string regex entry.")
        regex_name = "regex"
        regex_valid = re.fullmatch(_python_validation_regex(regex_pattern), stage_text) is not None
        if not regex_valid:
            validation_error = f"Generated {stage_role} did not match the stage regex contract."
    return StageOutput(
        role=stage_role,
        text=stage_text,
        structured_output=structured_output,
        regex_name=regex_name,
        regex_valid=regex_valid,
        validation_error=validation_error,
    )


def _generate_stage_text(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_role: str,
    stage_max_tokens: Mapping[str, int],
    structured_output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> str:
    return _generate_stage_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        prompt=prompt,
        completed_stages=completed_stages,
        stage_role=stage_role,
        stage_max_tokens=stage_max_tokens,
        structured_output_overrides=structured_output_overrides,
        stage_text_chunk_callback=stage_text_chunk_callback,
    ).text


def _generate_stage_text_with_metrics(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_role: str,
    stage_max_tokens: Mapping[str, int],
    use_guidance: bool = True,
    seed: int | None = None,
    structured_output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> _StageGenerationResult:
    if bundle.profile.name == "24b" and stage_role == "book_plan":
        from . import orchestrator_24b

        return orchestrator_24b.generate_book_plan_text_with_metrics(
            bundle=bundle,
            runtime=runtime,
            prompt=prompt,
            completed_stages=completed_stages,
            stage_max_tokens=stage_max_tokens,
            use_guidance=use_guidance,
            seed=seed,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
    if _uses_preloaded_book_plan_workflow(bundle) and stage_role == "book_plan":
        from . import orchestrator_14b

        return orchestrator_14b.generate_book_plan_text_with_metrics(
            bundle=bundle,
            runtime=runtime,
            prompt=prompt,
            completed_stages=completed_stages,
            stage_max_tokens=stage_max_tokens,
            use_guidance=use_guidance,
            seed=seed,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
    rendered_prompt = bundle.protocol.render_stage_prompt(prompt, completed_stages, stage_role)
    structured_outputs = (
        _resolve_stage_structured_outputs(bundle, stage_role, structured_output_overrides) if use_guidance else None
    )
    return _generate_rendered_prompt_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        rendered_prompt=rendered_prompt,
        stage_role=stage_role,
        stage_max_tokens=stage_max_tokens,
        structured_outputs=structured_outputs,
        seed=seed,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )


def _generate_rendered_prompt_text_with_metrics(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    rendered_prompt: str,
    stage_role: str,
    stage_max_tokens: Mapping[str, int],
    structured_outputs: dict[str, Any] | None,
    stop_strings: list[str] | None = None,
    seed: int | None = None,
    temperature_override: float | None = None,
    top_k_override: int | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
    strip_leading_blank_lines: bool = True,
) -> _StageGenerationResult:
    prompt_tokens = bundle.tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    if not isinstance(prompt_tokens, list):
        raise TypeError("Tokenizer input_ids for rendered prompt must be a list.")
    available_completion_tokens = runtime.max_model_len - len(prompt_tokens) - REQUEST_MAX_TOKEN_SAFETY_MARGIN
    if available_completion_tokens <= 0:
        raise ValueError(
            f"Rendered {stage_role} prompt uses {len(prompt_tokens)} tokens, "
            f"which exceeds the runtime max length {runtime.max_model_len}."
        )
    requested_max_tokens = min(stage_max_tokens[stage_role], available_completion_tokens)
    generation_kwargs: dict[str, Any] = {
        "prompt": rendered_prompt,
        "max_tokens": requested_max_tokens,
        "temperature": _stage_temperature(stage_role) if temperature_override is None else temperature_override,
        "stop": bundle.protocol.stage_stop_strings(stage_role) if stop_strings is None else stop_strings,
        "structured_outputs": structured_outputs,
        "top_k": _stage_top_k(stage_role) if top_k_override is None else top_k_override,
        "seed": seed,
    }
    streamed_trimmed_text = ""
    streamed_raw_text = ""
    if stage_text_chunk_callback is not None:

        def stream_trimmed_stage_text_chunk(text_chunk: str) -> None:
            nonlocal streamed_raw_text, streamed_trimmed_text
            streamed_raw_text += text_chunk
            current_trimmed_text = bundle.protocol.trim_generated_stage_text(
                streamed_raw_text,
                stage_role,
                strip_leading_blank_lines=strip_leading_blank_lines,
            )
            if not current_trimmed_text.startswith(streamed_trimmed_text):
                raise ValueError("Trimmed streaming output stopped being a monotonic text extension.")
            text_delta = current_trimmed_text[len(streamed_trimmed_text) :]
            if text_delta:
                stage_text_chunk_callback(text_delta)
            streamed_trimmed_text = current_trimmed_text

        generation_kwargs["text_chunk_callback"] = stream_trimmed_stage_text_chunk
    raw_text = generate_completion(runtime, **generation_kwargs)
    trimmed_text = bundle.protocol.trim_generated_stage_text(
        raw_text,
        stage_role,
        strip_leading_blank_lines=strip_leading_blank_lines,
    )
    if stage_text_chunk_callback is not None:
        if not trimmed_text.startswith(streamed_trimmed_text):
            raise ValueError("Final trimmed output stopped being a monotonic text extension.")
        text_delta = trimmed_text[len(streamed_trimmed_text) :]
        if text_delta:
            stage_text_chunk_callback(text_delta)
    return _StageGenerationResult(
        text=trimmed_text,
        prompt_token_count=len(prompt_tokens),
        max_tokens_requested=requested_max_tokens,
    )


def _count_text_tokens(bundle: StoryBundle, text: str) -> int:
    text_tokens = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(text_tokens, list):
        raise TypeError("Tokenizer input_ids for generated text must be a list.")
    return len(text_tokens)


def _count_appended_text_tokens(bundle: StoryBundle, prompt_prefix: str, appended_text: str) -> int:
    return _count_text_tokens(bundle, prompt_prefix + appended_text) - _count_text_tokens(bundle, prompt_prefix)


def _ensure_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return text + "\n"


def _has_non_default_removed_stage_override(
    bundle: StoryBundle,
    stage_max_tokens: Mapping[str, int] | None,
    removed_stage_role: str,
) -> bool:
    if stage_max_tokens is None or removed_stage_role not in stage_max_tokens:
        return False
    default_stage_max_tokens = bundle.profile.default_stage_max_tokens
    if removed_stage_role not in default_stage_max_tokens:
        return True
    return stage_max_tokens[removed_stage_role] != default_stage_max_tokens[removed_stage_role]


def _append_stage_roles(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    stage_outputs: list[StageOutput],
    completed_stages: list[tuple[str, str]],
    stage_roles: tuple[str, ...],
    prompt: str,
    stage_max_tokens: Mapping[str, int],
    structured_output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    stage_output_callback: StageOutputCallback | None = None,
    stage_start_callback: StageStartCallback | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> str | None:
    for stage_role in stage_roles:
        if stage_start_callback is not None:
            stage_start_callback(stage_role)
        stage_text = _generate_stage_text(
            bundle=bundle,
            runtime=runtime,
            prompt=prompt,
            completed_stages=completed_stages,
            stage_role=stage_role,
            stage_max_tokens=stage_max_tokens,
            structured_output_overrides=structured_output_overrides,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        stage_output = _build_stage_output(bundle, stage_role, stage_text, structured_output_overrides)
        _record_stage_output(
            stage_outputs=stage_outputs,
            completed_stages=completed_stages,
            stage_output=stage_output,
            stage_output_callback=stage_output_callback,
        )
        if stage_output.regex_valid is False:
            return stage_role
    return None


def _record_stage_output(
    *,
    stage_outputs: list[StageOutput],
    completed_stages: list[tuple[str, str]],
    stage_output: StageOutput,
    stage_output_callback: StageOutputCallback | None = None,
) -> None:
    stage_outputs.append(stage_output)
    completed_stages.append((stage_output.role, stage_output.text))
    if stage_output_callback is not None:
        stage_output_callback(stage_output)


def _get_stage_text(stage_outputs: list[StageOutput], stage_role: str) -> str:
    for stage_output in stage_outputs:
        if stage_output.role == stage_role:
            return stage_output.text
    raise ValueError(f"Missing generated stage output for {stage_role!r}.")


def _chapter_text_prefill(chapter_name: str) -> str:
    return f"### {chapter_name}\n```\n"


def _merge_prefilled_stage_text(stage_prefill: str, generated_stage_text: str) -> str:
    if generated_stage_text.startswith(stage_prefill):
        return generated_stage_text
    stripped_generated_stage_text = generated_stage_text.lstrip()
    if (
        stripped_generated_stage_text.startswith("## ")
        or stripped_generated_stage_text.startswith("### ")
        or stripped_generated_stage_text.startswith("```")
    ):
        raise ValueError("Generated text stage returned a conflicting scaffold instead of chapter-body continuation.")
    return stage_prefill + generated_stage_text


def _build_generation_run(
    *,
    bundle: StoryBundle,
    prompt: str,
    model_identifier: str,
    stage_outputs: list[StageOutput],
    later_chapter_count: int,
    failed_stage_role: str | None,
) -> GenerationRun:
    artifact_text = render_story_stage_artifact(bundle.protocol, prompt, stage_outputs)
    book_text = render_book_text(stage_outputs)
    return GenerationRun(
        profile=bundle.profile.name,
        model=model_identifier,
        prompt=prompt,
        stages=stage_outputs,
        later_chapter_count=later_chapter_count,
        validation_success=failed_stage_role is None,
        failed_stage_role=failed_stage_role,
        artifact_text=artifact_text,
        book_text=book_text,
    )


def _generate_full_book_chapters_plan_text_with_metrics(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    planned_chapters: list[PlannedChapterInfo],
    later_chapter_count: int,
    stage_max_tokens: Mapping[str, int],
    use_guidance: bool = True,
    seed_base: int | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> _StageGenerationResult:
    from . import orchestrator_14b

    return orchestrator_14b.generate_full_book_chapters_plan_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        prompt=prompt,
        completed_stages=completed_stages,
        planned_chapters=planned_chapters,
        later_chapter_count=later_chapter_count,
        stage_max_tokens=stage_max_tokens,
        use_guidance=use_guidance,
        seed_base=seed_base,
        stage_text_chunk_callback=stage_text_chunk_callback,
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
    if bundle.profile.name == "14b":
        from . import orchestrator_14b

        return orchestrator_14b.generate_first_chapter(
            bundle,
            prompt=prompt,
            stage_max_tokens=stage_max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )

    from . import orchestrator_24b

    return orchestrator_24b.generate_first_chapter(
        bundle,
        prompt=prompt,
        stage_max_tokens=stage_max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        stage_output_callback=stage_output_callback,
        stage_start_callback=stage_start_callback,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )


def generate_full_book(
    bundle: StoryBundle,
    *,
    prompt: str,
    stage_max_tokens: Mapping[str, int] | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    later_chapter_limit: int | None = None,
    stage_output_callback: StageOutputCallback | None = None,
    stage_start_callback: StageStartCallback | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> GenerationRun:
    if bundle.profile.name == "14b":
        from . import orchestrator_14b

        return orchestrator_14b.generate_full_book(
            bundle,
            prompt=prompt,
            stage_max_tokens=stage_max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            later_chapter_limit=later_chapter_limit,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )

    if later_chapter_limit is not None:
        raise ValueError("later_chapter_limit is only supported for the 14b full-book workflow.")
    return generate_first_chapter(
        bundle,
        prompt=prompt,
        stage_max_tokens=stage_max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        stage_output_callback=stage_output_callback,
        stage_start_callback=stage_start_callback,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )


def validate_run(
    bundle: StoryBundle,
    *,
    prompt: str,
    stage_max_tokens: Mapping[str, int] | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    later_chapter_limit: int | None = None,
    stage_output_callback: StageOutputCallback | None = None,
    stage_start_callback: StageStartCallback | None = None,
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> GenerationRun:
    return generate_full_book(
        bundle,
        prompt=prompt,
        stage_max_tokens=stage_max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        later_chapter_limit=later_chapter_limit,
        stage_output_callback=stage_output_callback,
        stage_start_callback=stage_start_callback,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )
