from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import artifacts_14b, artifacts_24b

EOT_TOKEN = "<|eot_id|>"
START_HEADER_TOKEN = "<|start_header_id|>"
STOP_HEADER_TOKEN = "<|stop_header_id|>"

STAGE_ROLES_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "24b": (
        "prompt",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_scene_breakdown",
        "first_chapter_text",
    ),
    "14b": (
        "prompt",
        "book_preview",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_text",
        "full_book_chapters_plan",
        "book_characters_list",
        "scene_breakdown",
        "chapter_text",
    ),
}

DEFAULT_STAGE_SEQUENCE_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "24b": (
        "book_plan",
        "first_chapter_plan",
        "first_chapter_scene_breakdown",
        "first_chapter_text",
    ),
    "14b": (
        "book_preview",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_text",
        "full_book_chapters_plan",
        "book_characters_list",
        "scene_breakdown",
        "chapter_text",
    ),
}

STRUCTURED_OUTPUTS_BY_PROFILE: dict[str, dict[str, dict[str, str]]] = {
    "24b": {
        "book_plan": {"regex": artifacts_24b.BOOK_PLAN_REGEX_24B},
        "first_chapter_plan": {"regex": artifacts_24b.FIRST_CHAPTER_PLAN_REGEX_24B},
        "first_chapter_scene_breakdown": {"regex": artifacts_24b.SCENE_BREAKDOWN_REGEX},
    },
    "14b": {
        "book_preview": {"regex": artifacts_14b.BOOK_PREVIEW_REGEX},
        "book_plan": {"regex": artifacts_14b.BOOK_PLAN_REGEX_14B},
        "first_chapter_plan": {"regex": artifacts_14b.FIRST_CHAPTER_PLAN_REGEX_14B},
        "scene_breakdown": {"regex": artifacts_14b.SCENE_BREAKDOWN_REGEX},
        "book_characters_list": {"regex": artifacts_14b.BOOK_CHARACTERS_LIST_REGEX},
    },
}


def _stage_header(role: str) -> str:
    return f"{START_HEADER_TOKEN}{role}{STOP_HEADER_TOKEN}"


@dataclass
class PagestormStageProtocol:
    PROFILE_NAME: str
    DEFAULT_STAGE_SEQUENCE: tuple[str, ...]
    STAGE_ROLES: tuple[str, ...]
    _structured_outputs: dict[str, dict[str, str]]

    EOT_TOKEN: str = EOT_TOKEN

    def stage_header(self, role: str) -> str:
        if role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {role}")
        return _stage_header(role)

    def stage_stop_strings(self, role: str) -> list[str]:
        if role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {role}")
        return [self.EOT_TOKEN]

    def stage_structured_outputs(self, role: str) -> dict[str, Any] | None:
        structured_outputs = self._structured_outputs.get(role)
        if structured_outputs is None:
            return None
        return dict(structured_outputs)

    def render_stage_prompt(
        self, prompt: str, completed_stages: list[tuple[str, str]] | None, next_stage_role: str
    ) -> str:
        if next_stage_role == "prompt":
            raise ValueError("prompt cannot be used as a generation stage")
        if next_stage_role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {next_stage_role}")
        rendered_parts = [f"{self.stage_header('prompt')}\n\n{prompt}{self.EOT_TOKEN}"]
        for role, content in completed_stages or []:
            if role == "prompt":
                raise ValueError("completed_stages must not include prompt")
            rendered_parts.append(f"{self.stage_header(role)}\n\n{content}{self.EOT_TOKEN}")
        rendered_parts.append(f"{self.stage_header(next_stage_role)}\n\n")
        return "".join(rendered_parts)

    def trim_generated_stage_text(
        self,
        text: str,
        generated_stage_role: str,
        *,
        strip_leading_blank_lines: bool = True,
    ) -> str:
        if generated_stage_role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {generated_stage_role}")
        trimmed_stage_text = text.split(self.EOT_TOKEN, 1)[0]
        normalized_lines = [line.rstrip() for line in trimmed_stage_text.split("\n")]
        if strip_leading_blank_lines:
            while normalized_lines and normalized_lines[0] == "":
                normalized_lines.pop(0)
        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()
        return "\n".join(normalized_lines)


def build_story_stage_protocol(profile_name: str) -> PagestormStageProtocol:
    if profile_name not in STAGE_ROLES_BY_PROFILE:
        raise ValueError(f"Unsupported profile: {profile_name}")
    return PagestormStageProtocol(
        PROFILE_NAME=profile_name,
        DEFAULT_STAGE_SEQUENCE=DEFAULT_STAGE_SEQUENCE_BY_PROFILE[profile_name],
        STAGE_ROLES=STAGE_ROLES_BY_PROFILE[profile_name],
        _structured_outputs=STRUCTURED_OUTPUTS_BY_PROFILE[profile_name],
    )
