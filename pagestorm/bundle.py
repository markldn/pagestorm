from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models import StoryBundle
from .protocol import build_story_stage_protocol
from .profiles import PROFILES, get_profile


RUNTIME_ASSET_PATTERNS = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer*",
    "special_tokens_map.json",
    "tokenizer.model",
    "tekken.json",
    "params.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def _resolve_bundle_directory(
    bundle_path: str | Path | None,
    repo_id: str | None,
    revision: str | None,
    default_repo_id: str,
) -> tuple[Path, str | None]:
    if bundle_path is not None and repo_id is not None:
        raise ValueError("Specify at most one of bundle_path or repo_id.")

    if bundle_path is not None:
        resolved_bundle_path = Path(bundle_path).expanduser().resolve()
        if not resolved_bundle_path.exists():
            raise FileNotFoundError(f"Bundle path does not exist: {resolved_bundle_path}")
        return resolved_bundle_path, None

    resolved_repo_id = repo_id or default_repo_id
    snapshot_path = snapshot_download(
        repo_id=resolved_repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=list(RUNTIME_ASSET_PATTERNS),
    )
    return Path(snapshot_path).resolve(), resolved_repo_id


def _infer_bundle_profile_name_from_repo_id(repo_id: str | None) -> str | None:
    if repo_id is None:
        return None
    for profile in PROFILES.values():
        if profile.default_repo_id == repo_id:
            return profile.name
    return None


def _read_bundle_profile_name(bundle_path: Path, source_repo_id: str | None) -> str:
    tokenizer_config_path = bundle_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json at {tokenizer_config_path}")
    tokenizer_config_payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    bundle_profile_name = tokenizer_config_payload.get("pagestorm_profile")
    if isinstance(bundle_profile_name, str):
        return bundle_profile_name
    if bundle_profile_name is not None:
        raise ValueError("tokenizer_config.json pagestorm_profile must be a string when present.")

    inferred_profile_name = _infer_bundle_profile_name_from_repo_id(source_repo_id)
    if inferred_profile_name is not None:
        return inferred_profile_name

    raise ValueError(
        "tokenizer_config.json must include a string pagestorm_profile field for local bundles and non-default repos."
    )


def load_story_bundle(
    *,
    profile_name: str,
    bundle_path: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> StoryBundle:
    requested_profile = get_profile(profile_name)
    resolved_bundle_path, source_repo_id = _resolve_bundle_directory(
        bundle_path,
        repo_id,
        revision,
        requested_profile.default_repo_id,
    )
    bundle_profile_name = _read_bundle_profile_name(resolved_bundle_path, source_repo_id)
    if bundle_profile_name != profile_name:
        raise ValueError(f"Bundle profile {bundle_profile_name!r} did not match requested profile {profile_name!r}.")
    tokenizer = AutoTokenizer.from_pretrained(resolved_bundle_path)
    return StoryBundle(
        bundle_path=resolved_bundle_path,
        tokenizer=tokenizer,
        protocol=build_story_stage_protocol(requested_profile.name),
        profile=requested_profile,
        source_repo_id=source_repo_id,
    )
