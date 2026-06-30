from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .profiles import ModelProfile


class StoryStageProtocol(Protocol):
    PROFILE_NAME: str
    EOT_TOKEN: str
    DEFAULT_STAGE_SEQUENCE: tuple[str, ...]
    STAGE_ROLES: tuple[str, ...]

    def stage_header(self, role: str) -> str: ...

    def stage_stop_strings(self, role: str) -> list[str]: ...

    def stage_structured_outputs(self, role: str) -> dict[str, Any] | None: ...

    def render_stage_prompt(
        self, prompt: str, completed_stages: list[tuple[str, str]] | None, next_stage_role: str
    ) -> str: ...

    def trim_generated_stage_text(
        self,
        text: str,
        generated_stage_role: str,
        *,
        strip_leading_blank_lines: bool = True,
    ) -> str: ...


@dataclass(frozen=True)
class StoryBundle:
    bundle_path: Path
    tokenizer: Any
    protocol: StoryStageProtocol
    profile: ModelProfile
    source_repo_id: str | None = None


StageOutputCallback = Callable[["StageOutput"], None]
StageStartCallback = Callable[[str], None]
StageTextChunkCallback = Callable[[str], None]


@dataclass(frozen=True)
class StageOutput:
    role: str
    text: str
    structured_output: dict[str, Any] | None = None
    regex_name: str | None = None
    regex_valid: bool | None = None
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "structured_output": self.structured_output,
            "regex_name": self.regex_name,
            "regex_valid": self.regex_valid,
            "validation_error": self.validation_error,
        }


@dataclass(frozen=True)
class GenerationRun:
    profile: str
    model: str
    prompt: str
    stages: list[StageOutput]
    later_chapter_count: int
    validation_success: bool
    failed_stage_role: str | None
    artifact_text: str
    book_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": self.model,
            "prompt": self.prompt,
            "validation_success": self.validation_success,
            "failed_stage_role": self.failed_stage_role,
            "later_chapter_count": self.later_chapter_count,
            "stages": [stage.to_dict() for stage in self.stages],
        }
