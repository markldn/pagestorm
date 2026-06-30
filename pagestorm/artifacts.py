from __future__ import annotations

import re
from pathlib import Path

from .models import GenerationRun, StageOutput, StoryStageProtocol


BOOK_TITLE_RE = re.compile(r"(?m)^## Book Title\n(?P<title>[^\n]+)")
UNSAFE_FILENAME_CHAR_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def render_story_stage_artifact(
    protocol: StoryStageProtocol,
    prompt: str,
    stage_outputs: list[StageOutput],
) -> str:
    artifact_parts = [
        f"{protocol.stage_header('prompt')}\n\n{prompt}{protocol.EOT_TOKEN}",
    ]
    for stage_output in stage_outputs:
        artifact_parts.append(
            f"{protocol.stage_header(stage_output.role)}\n\n"
            f"{stage_output.text}{protocol.EOT_TOKEN}"
        )
    return "".join(artifact_parts)


def render_book_text(stage_outputs: list[StageOutput]) -> str:
    chapter_text_blocks = [
        stage_output.text for stage_output in stage_outputs if stage_output.role in {"first_chapter_text", "chapter_text"}
    ]
    return "\n\n".join(chapter_text_blocks).strip()


def _extract_generated_book_title(stage_outputs: list[StageOutput]) -> str | None:
    for stage_output in stage_outputs:
        title_match = BOOK_TITLE_RE.search(stage_output.text)
        if title_match is not None:
            title = title_match.group("title").strip()
            if title:
                return title
    return None


def _sanitize_book_filename(title: str) -> str:
    sanitized_title = UNSAFE_FILENAME_CHAR_RE.sub(" ", title)
    sanitized_title = re.sub(r"\s+", " ", sanitized_title).strip(" .")
    if not sanitized_title:
        raise ValueError("Generated book title did not contain any filesystem-safe filename characters.")
    return f"{sanitized_title}.txt"


def _write_titled_book_text(output_directory: Path, stage_outputs: list[StageOutput], book_text: str) -> None:
    generated_book_title = _extract_generated_book_title(stage_outputs)
    if generated_book_title is None:
        return
    (output_directory / _sanitize_book_filename(generated_book_title)).write_text(book_text, encoding="utf-8")


def write_partial_generation_outputs(
    output_directory: str | Path,
    protocol: StoryStageProtocol,
    prompt: str,
    stage_outputs: list[StageOutput],
) -> None:
    resolved_output_directory = Path(output_directory).expanduser().resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    (resolved_output_directory / "artifact.txt").write_text(
        render_story_stage_artifact(protocol, prompt, stage_outputs),
        encoding="utf-8",
    )
    _write_titled_book_text(resolved_output_directory, stage_outputs, render_book_text(stage_outputs))


def write_generation_outputs(output_directory: str | Path, generation_run: GenerationRun) -> None:
    resolved_output_directory = Path(output_directory).expanduser().resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    (resolved_output_directory / "artifact.txt").write_text(generation_run.artifact_text, encoding="utf-8")
    _write_titled_book_text(resolved_output_directory, generation_run.stages, generation_run.book_text)
