from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_STAGE_MAX_TOKENS_24B: dict[str, int] = {
    "book_plan": 50_000,
    "first_chapter_plan": 9_000,
    "first_chapter_scene_breakdown": 3_000,
    "first_chapter_text": 12_000,
}

DEFAULT_STAGE_MAX_TOKENS_14B: dict[str, int] = {
    "book_preview": 4_000,
    "book_plan": 50_000,
    "first_chapter_plan": 9_000,
    "first_chapter_text": 12_000,
    "full_book_chapters_plan": 60_000,
    "book_characters_list": 6_000,
    "scene_breakdown": 3_000,
    "chapter_text": 16_000,
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    default_repo_id: str
    default_max_model_len: int
    supports_full_book: bool
    default_stage_max_tokens: Mapping[str, int]
    stage_max_token_caps: Mapping[str, int]
    tokenizer_mode: str | None = None

    def resolve_stage_max_tokens(self, overrides: Mapping[str, int] | None = None) -> dict[str, int]:
        resolved_stage_max_tokens = dict(self.default_stage_max_tokens)
        if overrides is not None:
            for stage_role, max_tokens in overrides.items():
                if stage_role not in resolved_stage_max_tokens:
                    raise ValueError(f"Unsupported stage max-token override: {stage_role}")
                resolved_stage_max_tokens[stage_role] = max_tokens
        for stage_role, token_cap in self.stage_max_token_caps.items():
            resolved_stage_max_tokens[stage_role] = min(resolved_stage_max_tokens[stage_role], token_cap)
        return resolved_stage_max_tokens


PROFILES: dict[str, ModelProfile] = {
    "24b": ModelProfile(
        name="24b",
        default_repo_id="Pageshift-Entertainment/pagestorm-research-preview-24b-first-chapter-only",
        default_max_model_len=32_768,
        supports_full_book=False,
        default_stage_max_tokens=DEFAULT_STAGE_MAX_TOKENS_24B,
        stage_max_token_caps={"first_chapter_plan": 2_500},
        tokenizer_mode="mistral",
    ),
    "14b": ModelProfile(
        name="14b",
        default_repo_id="Pageshift-Entertainment/pagestorm-research-preview-14b-full-book",
        default_max_model_len=262_144,
        supports_full_book=True,
        default_stage_max_tokens=DEFAULT_STAGE_MAX_TOKENS_14B,
        stage_max_token_caps={},
        tokenizer_mode="mistral",
    ),
}


def get_profile(profile_name: str) -> ModelProfile:
    try:
        return PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported profile: {profile_name}") from exc


def get_profile_by_repo_id(repo_id: str) -> ModelProfile:
    matching_profiles = [profile for profile in PROFILES.values() if profile.default_repo_id == repo_id]
    assert len(matching_profiles) == 1, f"Unsupported Pagestorm repo_id: {repo_id}"
    return matching_profiles[0]
