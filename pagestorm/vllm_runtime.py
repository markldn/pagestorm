"""llama.cpp-server backed drop-in replacement for pagestorm's vLLM runtime.

This module keeps the exact public surface the orchestrators rely on
(`StoryVllmRuntime`, `create_generation_runtime`, `generate_completion`,
`close_generation_runtime`) but routes generation to a running
`llama-server` (`/completion`) instead of an in-process vLLM engine.

Why: the model is a `ministral3` arch that runs cleanly as a GGUF under the
ROCm llama.cpp build, avoiding a vLLM-on-ROCm install. The model was trained
to emit the staged formats, so generation is best-effort (no regex-guided
decoding); `structured_outputs` regexes are passed through to the server only
if it advertises support, otherwise ignored. Strict regex enforcement still
happens in the orchestrator's own validation pass.

Connection: set `PAGESTORM_LLAMA_URL`, or set `PAGESTORM_LLAMA_HOST` /
`PAGESTORM_LLAMA_PORT` and let the URL be assembled at runtime.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .models import StoryBundle
from .regex_to_gbnf import regex_to_gbnf


DEFAULT_LLAMA_HOST = os.environ.get("PAGESTORM_LLAMA_HOST", "localhost")
DEFAULT_LLAMA_PORT = os.environ.get("PAGESTORM_LLAMA_PORT", "8091")
DEFAULT_SERVER_URL = os.environ.get("PAGESTORM_LLAMA_URL", f"http://{DEFAULT_LLAMA_HOST}:{DEFAULT_LLAMA_PORT}")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("PAGESTORM_LLAMA_TIMEOUT", "1800"))


class GenerationCancelled(Exception):
    """Raised inside generate_completion when the caller requests a stop."""


# Per-thread cancel flags. A generation runs in one worker thread; the caller
# registers a threading.Event for that thread and sets it to abort mid-stream.
_CANCEL: dict[int, threading.Event] = {}


def register_cancel(event: threading.Event) -> None:
    _CANCEL[threading.get_ident()] = event


def unregister_cancel() -> None:
    _CANCEL.pop(threading.get_ident(), None)


def _env_float(name: str):
    v = os.environ.get(name, "").strip()
    return float(v) if v else None


def _env_int(name: str):
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


@dataclass(frozen=True)
class StoryVllmRuntime:
    server_url: str
    max_model_len: int
    model_identifier: str


def _has_local_model_weights(bundle_path: Path) -> bool:
    if (bundle_path / "model.safetensors").exists():
        return True
    if (bundle_path / "model.safetensors.index.json").exists():
        return True
    return any(bundle_path.glob("*.safetensors")) or any(bundle_path.glob("*.gguf"))


def _resolve_runtime_model_identifier(bundle: StoryBundle) -> str:
    if bundle.source_repo_id is not None:
        return bundle.source_repo_id
    # The GGUF may live outside the bundle dir; identifier is informational only.
    return str(bundle.bundle_path)


def _wait_for_server(server_url: str) -> None:
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"llama-server is not reachable at {server_url}. Start it first "
            f"(see serve.sh). Underlying error: {exc}"
        ) from exc


def create_generation_runtime(
    bundle: StoryBundle,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> StoryVllmRuntime:
    # tensor_parallel_size / gpu_memory_utilization are vLLM concepts and are
    # intentionally ignored here; the llama-server process owns GPU placement.
    del tensor_parallel_size, gpu_memory_utilization
    # Read at call time so a caller (e.g. the GUI) can override per run.
    server_url = os.environ.get("PAGESTORM_LLAMA_URL", DEFAULT_SERVER_URL).rstrip("/")
    _wait_for_server(server_url)
    resolved_max_model_len = max_model_len or bundle.profile.default_max_model_len
    return StoryVllmRuntime(
        server_url=server_url,
        max_model_len=int(resolved_max_model_len),
        model_identifier=_resolve_runtime_model_identifier(bundle),
    )


def generate_completion(
    runtime: StoryVllmRuntime,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str],
    structured_outputs: dict[str, Any] | None,
    top_k: int | None = None,
    seed: int | None = None,
    text_chunk_callback: Any | None = None,
) -> str:
    # Optional global sampling overrides (set by the GUI). Blank/unset => use the
    # orchestrator's per-stage values (the tuned defaults).
    temp_override = _env_float("PAGESTORM_TEMPERATURE")
    topk_override = _env_int("PAGESTORM_TOP_K")
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": temp_override if temp_override is not None else float(temperature),
        "stop": list(stop),
        "stream": True,
        "cache_prompt": True,
    }
    effective_top_k = topk_override if topk_override is not None else top_k
    if effective_top_k is not None:
        payload["top_k"] = int(effective_top_k)
    if seed is not None:
        payload["seed"] = int(seed)
    top_p = _env_float("PAGESTORM_TOP_P")
    if top_p is not None:
        payload["top_p"] = top_p
    min_p = _env_float("PAGESTORM_MIN_P")
    if min_p is not None:
        payload["min_p"] = min_p
    rep = _env_float("PAGESTORM_REPEAT_PENALTY")
    if rep is not None:
        payload["repeat_penalty"] = rep
    model_name = os.environ.get("PAGESTORM_MODEL", "").strip()
    if model_name:
        payload["model"] = model_name  # used by routers/llama-swap; ignored by plain llama.cpp
    # Structured outputs: vLLM constrained decoding to a per-stage regex.
    # llama.cpp's /completion takes a GBNF `grammar`, so we compile the regex to
    # GBNF and send that. Enabled by default; set PAGESTORM_GRAMMAR=0 to disable
    # (falls back to best-effort, relying on the orchestrator's validation pass).
    # Any conversion failure also falls back rather than aborting the run.
    if (
        structured_outputs is not None
        and structured_outputs.get("regex")
        and os.environ.get("PAGESTORM_GRAMMAR", "1") != "0"
    ):
        try:
            payload["grammar"] = regex_to_gbnf(structured_outputs["regex"])
        except Exception as exc:  # noqa: BLE001 - never let grammar break a run
            print(f"[pagestorm] regex->GBNF failed, generating unconstrained: {exc}", flush=True)

    cancel = _CANCEL.get(threading.get_ident())
    streamed_text = ""
    with requests.post(
        f"{runtime.server_url}/completion",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        # Iterate raw bytes and parse as UTF-8 ourselves. `decode_unicode=True`
        # would use the response's apparent encoding, which defaults to latin-1
        # for SSE (no charset header) and double-encodes non-ASCII text.
        for raw_line in response.iter_lines():
            if cancel is not None and cancel.is_set():
                # Closing the streamed response stops llama.cpp generation.
                raise GenerationCancelled("generation stopped by user")
            if not raw_line:
                continue
            if raw_line.startswith(b"data:"):
                raw_line = raw_line[len(b"data:"):].strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)  # json.loads decodes bytes as UTF-8
            except json.JSONDecodeError:
                continue
            content_piece = event.get("content")
            if isinstance(content_piece, str) and content_piece:
                if text_chunk_callback is not None:
                    text_chunk_callback(content_piece)
                streamed_text += content_piece
            if event.get("stop") is True:
                break
    return streamed_text


def close_generation_runtime(runtime: StoryVllmRuntime) -> None:
    # The llama-server lifecycle is managed externally (serve.sh); nothing to
    # tear down per-run.
    del runtime
