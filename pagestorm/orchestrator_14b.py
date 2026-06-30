from __future__ import annotations

import math
from typing import Any, Mapping

from .artifacts_14b import (
    ARC_CHAPTER_COUNT_RE,
    BULLET_LINE_RE,
    EMBEDDING_SPACE_RE,
    PlannedChapterInfo,
    build_fourteen_b_book_plan_step_regexes,
    build_full_book_chapter_plan_continuation_regex,
    build_full_book_chapters_plan_regex,
    extract_arc_chapter_counts_from_book_plan,
    extract_arc_chapter_names_from_book_plan,
    extract_arc_titles_from_book_plan,
    extract_preloaded_chapters_from_book_plan,
)
from .models import GenerationRun, StageOutput, StageOutputCallback, StageStartCallback, StageTextChunkCallback, StoryBundle
from .orchestrator import (
    CHAPTER_NAMES_TEMPERATURE,
    _StageGenerationResult,
    _append_stage_roles,
    _build_generation_run,
    _build_stage_output,
    _chapter_text_prefill,
    _count_appended_text_tokens,
    _count_text_tokens,
    _ensure_supported_full_book_workflow,
    _escape_regex_literal,
    _first_chapter_stage_order,
    _generate_rendered_prompt_text_with_metrics,
    _get_stage_text,
    _merge_prefilled_stage_text,
    _record_stage_output,
    _uses_full_book_workflow,
)
from .vllm_runtime import StoryVllmRuntime, close_generation_runtime


def _full_book_chapter_plan_prefill(
    planned_chapter: PlannedChapterInfo,
    *,
    include_leading_separator: bool,
) -> str:
    leading_separator = "\n\n" if include_leading_separator else ""
    return (
        f"{leading_separator}### {planned_chapter['chapter_name']}\n"
        f"**Word Count:** {planned_chapter['chapter_word_count']}\n"
        f"**Embedding Space:** {planned_chapter['chapter_embedding_space']}"
    )


def _inverse_word_count_upper_bound(rounded_word_count: int) -> int:
    def calculate_upper_bound(
        *,
        step: int,
        lower_bound: int,
        upper_bound: float,
        margin: float,
    ) -> int | None:
        if rounded_word_count % step != 0:
            return None

        quotient = rounded_word_count // step
        if quotient % 2 == 0:
            candidate_upper_bound = math.floor((quotient + 0.5) * step)
        else:
            candidate_upper_bound = math.ceil((quotient + 0.5) * step) - 1

        candidate_upper_bound = min(candidate_upper_bound, upper_bound)
        if candidate_upper_bound < lower_bound:
            return None
        return math.ceil(candidate_upper_bound * (1 + margin))

    candidates = [
        calculate_upper_bound(step=10, lower_bound=0, upper_bound=99, margin=0.50),
        calculate_upper_bound(step=50, lower_bound=100, upper_bound=999, margin=0.25),
        calculate_upper_bound(step=100, lower_bound=1000, upper_bound=9999, margin=0.15),
        calculate_upper_bound(step=500, lower_bound=10000, upper_bound=float("inf"), margin=0.08),
    ]
    resolved_candidates = [candidate for candidate in candidates if candidate is not None]
    if not resolved_candidates:
        raise ValueError(f"Chapter word count {rounded_word_count} does not match the expected rounded count bins.")
    return max(resolved_candidates)


def _chapter_text_target_max_tokens(chapter_word_count: int) -> int:
    return _inverse_word_count_upper_bound(chapter_word_count) * 2


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
    accumulated_stage_text = ""
    latest_generation_result: _StageGenerationResult | None = None
    default_stop_strings = list(bundle.protocol.stage_stop_strings("book_plan"))
    next_seed_offset = 0
    total_max_tokens_requested = 0

    def append_guided_book_plan_step(
        *,
        step_regex: str,
        next_section_header: str | None,
        temperature_override: float | None = None,
    ) -> None:
        nonlocal accumulated_stage_text, latest_generation_result, next_seed_offset, total_max_tokens_requested
        remaining_stage_tokens = stage_max_tokens["book_plan"] - _count_appended_text_tokens(
            bundle,
            rendered_prompt,
            accumulated_stage_text,
        )
        if remaining_stage_tokens <= 0:
            raise ValueError("14b book_plan exhausted the configured stage token budget before completing all sections.")
        step_stop_strings = list(default_stop_strings)
        if next_section_header is not None:
            step_stop_strings.insert(0, next_section_header)
        latest_generation_result = _generate_rendered_prompt_text_with_metrics(
            bundle=bundle,
            runtime=runtime,
            rendered_prompt=rendered_prompt + accumulated_stage_text,
            stage_role="book_plan",
            stage_max_tokens={"book_plan": remaining_stage_tokens},
            structured_outputs=None if not use_guidance else {"regex": step_regex},
            stop_strings=step_stop_strings,
            seed=None if seed is None else seed + next_seed_offset,
            temperature_override=temperature_override,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        next_seed_offset += 1
        total_max_tokens_requested += latest_generation_result.max_tokens_requested
        accumulated_stage_text += latest_generation_result.text
        if next_section_header is not None:
            accumulated_stage_text = accumulated_stage_text.rstrip("\n") + "\n\n"
            if stage_text_chunk_callback is not None:
                stage_text_chunk_callback("\n\n")

    append_guided_book_plan_step(
        step_regex=build_fourteen_b_book_plan_step_regexes()[0],
        next_section_header="## Medium Story Arcs",
    )
    short_story_arc_titles = extract_arc_titles_from_book_plan(accumulated_stage_text, "Short Story Arcs")

    def append_per_arc_section(
        *,
        arc_titles: list[str],
        arc_step_regexes: list[str],
        next_section_header: str | None,
        temperature_override: float | None = None,
    ) -> None:
        for arc_index, arc_step_regex in enumerate(arc_step_regexes):
            next_arc_header = None
            if arc_index + 1 < len(arc_step_regexes):
                next_arc_header = f"### {arc_titles[arc_index + 1]}"
            append_guided_book_plan_step(
                step_regex=arc_step_regex,
                next_section_header=next_arc_header if next_arc_header is not None else next_section_header,
                temperature_override=temperature_override,
            )

    medium_arc_regexes = [
        (
            r"^"
            + (r"## Medium Story Arcs\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n"
            + rf"(?:{BULLET_LINE_RE}\n){{1,40}}\n?"
            + r"$"
        )
        for arc_index, arc_title in enumerate(short_story_arc_titles)
    ]
    append_per_arc_section(
        arc_titles=short_story_arc_titles,
        arc_step_regexes=medium_arc_regexes,
        next_section_header="## Long Story Arcs",
    )

    long_arc_regexes = [
        (
            r"^"
            + (r"## Long Story Arcs\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n"
            + rf"(?:{BULLET_LINE_RE}\n){{1,80}}\n?"
            + r"$"
        )
        for arc_index, arc_title in enumerate(short_story_arc_titles)
    ]
    append_per_arc_section(
        arc_titles=short_story_arc_titles,
        arc_step_regexes=long_arc_regexes,
        next_section_header="## Number of Chapters",
    )

    chapter_count_regexes = [
        (
            r"^"
            + (r"## Number of Chapters\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n\* "
            + ARC_CHAPTER_COUNT_RE
            + r"\n?"
            + r"$"
        )
        for arc_index, arc_title in enumerate(short_story_arc_titles)
    ]
    append_per_arc_section(
        arc_titles=short_story_arc_titles,
        arc_step_regexes=chapter_count_regexes,
        next_section_header="## Chapter Names",
    )

    arc_chapter_counts = extract_arc_chapter_counts_from_book_plan(accumulated_stage_text)
    chapter_name_regexes = [
        (
            r"^"
            + (r"## Chapter Names\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n"
            + rf"(?:- [^\n]{{1,120}}\n){{{chapter_count}}}\n?"
            + r"$"
        )
        for arc_index, (arc_title, chapter_count) in enumerate(arc_chapter_counts)
    ]
    append_per_arc_section(
        arc_titles=[arc_title for arc_title, _ in arc_chapter_counts],
        arc_step_regexes=chapter_name_regexes,
        next_section_header="## Chapters Embedding Space",
        temperature_override=CHAPTER_NAMES_TEMPERATURE,
    )

    arc_chapter_names = extract_arc_chapter_names_from_book_plan(accumulated_stage_text)
    embedding_space_regexes = []
    for arc_index, (arc_title, chapter_names) in enumerate(arc_chapter_names):
        chapter_embedding_patterns = "".join(
            r"#### " + _escape_regex_literal(chapter_name) + r"\n" + EMBEDDING_SPACE_RE + r"\n"
            for chapter_name in chapter_names
        )
        embedding_space_regexes.append(
            r"^"
            + (r"## Chapters Embedding Space\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n"
            + chapter_embedding_patterns
            + r"\n?$"
        )
    append_per_arc_section(
        arc_titles=[arc_title for arc_title, _ in arc_chapter_names],
        arc_step_regexes=embedding_space_regexes,
        next_section_header="## Chapters Word Count",
    )

    word_count_regexes = []
    for arc_index, (arc_title, chapter_names) in enumerate(arc_chapter_names):
        chapter_word_count_patterns = "".join(
            r"#### " + _escape_regex_literal(chapter_name) + r"\n[0-9]{2,5}\n"
            for chapter_name in chapter_names
        )
        word_count_regexes.append(
            r"^"
            + (r"## Chapters Word Count\n" if arc_index == 0 else "")
            + r"### "
            + _escape_regex_literal(arc_title)
            + r"\n"
            + chapter_word_count_patterns
            + r"\n?$"
        )
    append_per_arc_section(
        arc_titles=[arc_title for arc_title, _ in arc_chapter_names],
        arc_step_regexes=word_count_regexes,
        next_section_header=None,
    )

    assert latest_generation_result is not None, "14b book_plan guidance steps must not be empty."
    return _StageGenerationResult(
        text=accumulated_stage_text,
        prompt_token_count=latest_generation_result.prompt_token_count,
        max_tokens_requested=total_max_tokens_requested,
    )


def generate_full_book_chapters_plan_text_with_metrics(
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
    rendered_prompt = bundle.protocol.render_stage_prompt(prompt, completed_stages, "full_book_chapters_plan")
    accumulated_stage_text = ""
    latest_generation_result: _StageGenerationResult | None = None
    total_max_tokens_requested = 0

    for later_chapter_index, planned_chapter in enumerate(planned_chapters[1 : later_chapter_count + 1]):
        chapter_plan_prefill = _full_book_chapter_plan_prefill(
            planned_chapter,
            include_leading_separator=later_chapter_index > 0,
        )
        prefilled_stage_token_count = _count_appended_text_tokens(
            bundle,
            rendered_prompt + accumulated_stage_text,
            chapter_plan_prefill,
        )
        prefilled_stage_text = accumulated_stage_text + chapter_plan_prefill
        remaining_stage_tokens = stage_max_tokens["full_book_chapters_plan"] - _count_appended_text_tokens(
            bundle,
            rendered_prompt,
            prefilled_stage_text,
        )
        if remaining_stage_tokens <= 0:
            raise ValueError(
                "14b full_book_chapters_plan exhausted the configured stage token budget before completing all chapters."
            )
        if stage_text_chunk_callback is not None:
            stage_text_chunk_callback(chapter_plan_prefill)
        chapter_step_structured_outputs = None
        chapter_step_stop_strings = list(bundle.protocol.stage_stop_strings("full_book_chapters_plan"))
        next_later_chapter_index = later_chapter_index + 2
        if next_later_chapter_index <= later_chapter_count:
            next_chapter_name = planned_chapters[next_later_chapter_index]["chapter_name"]
            if next_chapter_name != planned_chapter["chapter_name"]:
                chapter_step_stop_strings.insert(0, f"### {next_chapter_name}")
        if use_guidance:
            chapter_step_structured_outputs = {"regex": build_full_book_chapter_plan_continuation_regex()}
        latest_generation_result = _generate_rendered_prompt_text_with_metrics(
            bundle=bundle,
            runtime=runtime,
            rendered_prompt=rendered_prompt + prefilled_stage_text,
            stage_role="full_book_chapters_plan",
            stage_max_tokens={"full_book_chapters_plan": remaining_stage_tokens},
            structured_outputs=chapter_step_structured_outputs,
            stop_strings=chapter_step_stop_strings,
            seed=None if seed_base is None else seed_base + later_chapter_index,
            stage_text_chunk_callback=stage_text_chunk_callback,
            strip_leading_blank_lines=False,
        )
        total_max_tokens_requested += latest_generation_result.max_tokens_requested + prefilled_stage_token_count
        accumulated_stage_text = prefilled_stage_text + latest_generation_result.text

    if latest_generation_result is None:
        raise ValueError("Expected at least one later chapter when generating the 14b chapter-plan continuation.")

    return _StageGenerationResult(
        text=accumulated_stage_text,
        prompt_token_count=latest_generation_result.prompt_token_count,
        max_tokens_requested=total_max_tokens_requested,
    )


def _generate_text_stage_with_metrics(
    *,
    bundle: StoryBundle,
    runtime: StoryVllmRuntime,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_role: str,
    chapter_name: str,
    chapter_word_count: int,
    stage_max_tokens: Mapping[str, int],
    stage_text_chunk_callback: StageTextChunkCallback | None = None,
) -> _StageGenerationResult:
    rendered_prompt = bundle.protocol.render_stage_prompt(prompt, completed_stages, stage_role)
    stage_prefill = _chapter_text_prefill(chapter_name)
    prefilled_stage_token_count = _count_text_tokens(bundle, stage_prefill)
    requested_chapter_text_tokens = min(
        stage_max_tokens[stage_role],
        _chapter_text_target_max_tokens(chapter_word_count),
    )
    if requested_chapter_text_tokens <= 0:
        raise ValueError(f"{stage_role} token budget must be positive.")
    if stage_text_chunk_callback is not None:
        stage_text_chunk_callback(stage_prefill)
    generation_result = _generate_rendered_prompt_text_with_metrics(
        bundle=bundle,
        runtime=runtime,
        rendered_prompt=rendered_prompt + stage_prefill,
        stage_role=stage_role,
        stage_max_tokens={**stage_max_tokens, stage_role: requested_chapter_text_tokens},
        structured_outputs=None,
        stage_text_chunk_callback=stage_text_chunk_callback,
    )
    return _StageGenerationResult(
        text=_merge_prefilled_stage_text(stage_prefill, generation_result.text),
        prompt_token_count=generation_result.prompt_token_count,
        max_tokens_requested=generation_result.max_tokens_requested + prefilled_stage_token_count,
    )


def _first_chapter_prelude_stage_roles(bundle: StoryBundle) -> tuple[str, ...]:
    first_chapter_stage_roles = _first_chapter_stage_order(bundle)
    if "first_chapter_text" not in first_chapter_stage_roles:
        raise ValueError("14b first-chapter workflow must include first_chapter_text.")
    return tuple(stage_role for stage_role in first_chapter_stage_roles if stage_role != "first_chapter_text")


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
            stage_roles=_first_chapter_prelude_stage_roles(bundle),
            prompt=prompt,
            stage_max_tokens=resolved_stage_max_tokens,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        if failed_stage_role is None:
            try:
                first_planned_chapter = extract_preloaded_chapters_from_book_plan(_get_stage_text(stage_outputs, "book_plan"))[0]
            except (IndexError, ValueError):
                failed_stage_role = "book_plan"
            else:
                if stage_start_callback is not None:
                    stage_start_callback("first_chapter_text")
                first_chapter_text_result = _generate_text_stage_with_metrics(
                    bundle=bundle,
                    runtime=runtime,
                    prompt=prompt,
                    completed_stages=completed_stages,
                    stage_role="first_chapter_text",
                    chapter_name=first_planned_chapter["chapter_name"],
                    chapter_word_count=first_planned_chapter["chapter_word_count"],
                    stage_max_tokens=resolved_stage_max_tokens,
                    stage_text_chunk_callback=stage_text_chunk_callback,
                )
                first_chapter_text_output = _build_stage_output(
                    bundle,
                    "first_chapter_text",
                    first_chapter_text_result.text,
                )
                _record_stage_output(
                    stage_outputs=stage_outputs,
                    completed_stages=completed_stages,
                    stage_output=first_chapter_text_output,
                    stage_output_callback=stage_output_callback,
                )
                if first_chapter_text_output.regex_valid is False:
                    failed_stage_role = "first_chapter_text"
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
    from . import orchestrator

    if not _uses_full_book_workflow(bundle):
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
    _ensure_supported_full_book_workflow(bundle)

    if later_chapter_limit is not None and later_chapter_limit <= 0:
        raise ValueError("later_chapter_limit must be positive when provided.")

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
            stage_roles=_first_chapter_prelude_stage_roles(bundle),
            prompt=prompt,
            stage_max_tokens=resolved_stage_max_tokens,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        if failed_stage_role is not None:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=0,
                failed_stage_role=failed_stage_role,
            )

        try:
            planned_chapters = extract_preloaded_chapters_from_book_plan(_get_stage_text(stage_outputs, "book_plan"))
        except ValueError:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=0,
                failed_stage_role="book_plan",
            )
        if len(planned_chapters) < 2:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=0,
                failed_stage_role="book_plan",
            )

        if stage_start_callback is not None:
            stage_start_callback("first_chapter_text")
        first_chapter_text_result = _generate_text_stage_with_metrics(
            bundle=bundle,
            runtime=runtime,
            prompt=prompt,
            completed_stages=completed_stages,
            stage_role="first_chapter_text",
            chapter_name=planned_chapters[0]["chapter_name"],
            chapter_word_count=planned_chapters[0]["chapter_word_count"],
            stage_max_tokens=resolved_stage_max_tokens,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        first_chapter_text_output = _build_stage_output(
            bundle,
            "first_chapter_text",
            first_chapter_text_result.text,
        )
        _record_stage_output(
            stage_outputs=stage_outputs,
            completed_stages=completed_stages,
            stage_output=first_chapter_text_output,
            stage_output_callback=stage_output_callback,
        )
        if first_chapter_text_output.regex_valid is False:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=0,
                failed_stage_role="first_chapter_text",
            )

        later_chapter_count = len(planned_chapters) - 1
        if later_chapter_limit is not None:
            later_chapter_count = min(later_chapter_count, later_chapter_limit)
        if later_chapter_count == 0:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=0,
                failed_stage_role=None,
            )

        selected_planned_chapters = planned_chapters[: later_chapter_count + 1]
        full_book_structured_output_overrides: Mapping[str, Mapping[str, Any]] = {
            "full_book_chapters_plan": {
                "regex": build_full_book_chapters_plan_regex(selected_planned_chapters),
            }
        }
        if stage_start_callback is not None:
            stage_start_callback("full_book_chapters_plan")
        full_book_chapters_plan_text = generate_full_book_chapters_plan_text_with_metrics(
            bundle=bundle,
            runtime=runtime,
            prompt=prompt,
            completed_stages=completed_stages,
            planned_chapters=selected_planned_chapters,
            later_chapter_count=later_chapter_count,
            stage_max_tokens=resolved_stage_max_tokens,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        full_book_chapters_plan_output = _build_stage_output(
            bundle,
            "full_book_chapters_plan",
            full_book_chapters_plan_text.text,
            full_book_structured_output_overrides,
        )
        _record_stage_output(
            stage_outputs=stage_outputs,
            completed_stages=completed_stages,
            stage_output=full_book_chapters_plan_output,
            stage_output_callback=stage_output_callback,
        )
        if full_book_chapters_plan_output.regex_valid is False:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=later_chapter_count,
                failed_stage_role="full_book_chapters_plan",
            )

        failed_stage_role = _append_stage_roles(
            bundle=bundle,
            runtime=runtime,
            stage_outputs=stage_outputs,
            completed_stages=completed_stages,
            stage_roles=("book_characters_list",),
            prompt=prompt,
            stage_max_tokens=resolved_stage_max_tokens,
            structured_output_overrides=full_book_structured_output_overrides,
            stage_output_callback=stage_output_callback,
            stage_start_callback=stage_start_callback,
            stage_text_chunk_callback=stage_text_chunk_callback,
        )
        if failed_stage_role is not None:
            return _build_generation_run(
                bundle=bundle,
                prompt=prompt,
                model_identifier=runtime.model_identifier,
                stage_outputs=stage_outputs,
                later_chapter_count=later_chapter_count,
                failed_stage_role=failed_stage_role,
            )
        for planned_chapter in selected_planned_chapters[1:]:
            failed_stage_role = _append_stage_roles(
                bundle=bundle,
                runtime=runtime,
                stage_outputs=stage_outputs,
                completed_stages=completed_stages,
                stage_roles=("scene_breakdown",),
                prompt=prompt,
                stage_max_tokens=resolved_stage_max_tokens,
                stage_output_callback=stage_output_callback,
                stage_start_callback=stage_start_callback,
                stage_text_chunk_callback=stage_text_chunk_callback,
            )
            if failed_stage_role is not None:
                break

            if stage_start_callback is not None:
                stage_start_callback("chapter_text")
            chapter_text_result = _generate_text_stage_with_metrics(
                bundle=bundle,
                runtime=runtime,
                prompt=prompt,
                completed_stages=completed_stages,
                stage_role="chapter_text",
                chapter_name=planned_chapter["chapter_name"],
                chapter_word_count=planned_chapter["chapter_word_count"],
                stage_max_tokens=resolved_stage_max_tokens,
                stage_text_chunk_callback=stage_text_chunk_callback,
            )
            chapter_text_output = _build_stage_output(
                bundle,
                "chapter_text",
                chapter_text_result.text,
            )
            _record_stage_output(
                stage_outputs=stage_outputs,
                completed_stages=completed_stages,
                stage_output=chapter_text_output,
                stage_output_callback=stage_output_callback,
            )
            if chapter_text_output.regex_valid is False:
                failed_stage_role = "chapter_text"
                break

        return _build_generation_run(
            bundle=bundle,
            prompt=prompt,
            model_identifier=runtime.model_identifier,
            stage_outputs=stage_outputs,
            later_chapter_count=later_chapter_count,
            failed_stage_role=failed_stage_role,
        )
    finally:
        close_generation_runtime(runtime)
