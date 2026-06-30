from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from .models import StoryBundle


@dataclass(frozen=True)
class StoryVllmRuntime:
    llm: Any
    max_model_len: int
    model_identifier: str
    event_loop: Any | None = None
    event_loop_thread: Any | None = None


def _has_local_model_weights(bundle_path: Path) -> bool:
    if (bundle_path / "model.safetensors").exists():
        return True
    if (bundle_path / "model.safetensors.index.json").exists():
        return True
    return any(bundle_path.glob("*.safetensors"))


def _resolve_runtime_model_identifier(bundle: StoryBundle) -> str:
    if bundle.source_repo_id is not None:
        return bundle.source_repo_id
    if not _has_local_model_weights(bundle.bundle_path):
        raise FileNotFoundError(
            f"Local bundle path {bundle.bundle_path} does not contain model weights. "
            "Use --repo-id or a full local bundle export."
        )
    return str(bundle.bundle_path)


def _load_async_engine_runtime_types() -> tuple[Any, Any]:
    try:
        vllm_module = import_module("vllm")
    except ImportError as exc:
        raise RuntimeError("pagestorm requires the vllm Python package for generation.") from exc
    async_engine_class = getattr(vllm_module, "AsyncLLMEngine", None)
    if async_engine_class is None:
        try:
            async_engine_module = import_module("vllm.engine.async_llm_engine")
        except ImportError:
            async_engine_module = None
        if async_engine_module is not None:
            async_engine_class = getattr(async_engine_module, "AsyncLLMEngine", None)
    if async_engine_class is None:
        try:
            async_engine_module = import_module("vllm.v1.engine.async_llm")
        except ImportError:
            async_engine_module = None
        if async_engine_module is not None:
            async_engine_class = getattr(async_engine_module, "AsyncLLM", None)
    if async_engine_class is None:
        raise RuntimeError("pagestorm requires a vLLM async engine implementation for in-process streaming generation.")

    async_engine_args_class = getattr(vllm_module, "AsyncEngineArgs", None)
    if async_engine_args_class is None:
        try:
            async_engine_args_module = import_module("vllm.engine.arg_utils")
        except ImportError as exc:
            raise RuntimeError("pagestorm requires vllm.engine.arg_utils.AsyncEngineArgs for generation.") from exc
        async_engine_args_class = getattr(async_engine_args_module, "AsyncEngineArgs", None)
    if async_engine_args_class is None:
        raise RuntimeError("pagestorm requires vLLM AsyncEngineArgs for in-process generation.")
    return async_engine_class, async_engine_args_class


def _load_sampling_params_types() -> tuple[Any, Any]:
    try:
        sampling_params_module = import_module("vllm.sampling_params")
    except ImportError as exc:
        raise RuntimeError("pagestorm requires vllm.sampling_params for generation.") from exc
    return getattr(sampling_params_module, "SamplingParams"), getattr(sampling_params_module, "StructuredOutputsParams")


def _run_in_event_loop(event_loop: asyncio.AbstractEventLoop, coroutine: Any) -> Any:
    return asyncio.run_coroutine_threadsafe(coroutine, event_loop).result()


def _run_event_loop_forever(event_loop: asyncio.AbstractEventLoop, loop_started: threading.Event) -> None:
    asyncio.set_event_loop(event_loop)
    loop_started.set()
    event_loop.run_forever()


async def _drain_pending_event_loop_tasks() -> None:
    current_task = asyncio.current_task()
    pending_tasks = [task for task in asyncio.all_tasks() if task is not current_task and not task.done()]
    for pending_task in pending_tasks:
        pending_task.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _call_sync_function(function: Any) -> None:
    function()


def _shutdown_event_loop_thread(
    event_loop: asyncio.AbstractEventLoop,
    event_loop_thread: threading.Thread | None,
) -> None:
    if not event_loop.is_closed():
        event_loop.call_soon_threadsafe(event_loop.stop)
    if event_loop_thread is not None and event_loop_thread.is_alive():
        event_loop_thread.join()
    if not event_loop.is_closed():
        event_loop.close()


def create_generation_runtime(
    bundle: StoryBundle,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> StoryVllmRuntime:
    async_engine_class, async_engine_args_class = _load_async_engine_runtime_types()
    model_identifier = _resolve_runtime_model_identifier(bundle)
    runtime_kwargs: dict[str, Any] = {
        "model": model_identifier,
        "tokenizer": str(bundle.bundle_path),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len or bundle.profile.default_max_model_len,
        "dtype": "float16",
        "structured_outputs_config": {"backend": "guidance"},
    }
    if bundle.profile.tokenizer_mode is not None:
        runtime_kwargs["tokenizer_mode"] = bundle.profile.tokenizer_mode
    event_loop = asyncio.new_event_loop()
    loop_started = threading.Event()
    event_loop_thread = threading.Thread(
        target=_run_event_loop_forever,
        args=(event_loop, loop_started),
        name="pagestorm-vllm-loop",
        daemon=True,
    )
    event_loop_thread.start()
    loop_started.wait()
    try:
        async_engine_args = async_engine_args_class(**runtime_kwargs)
        llm = _run_in_event_loop(
            event_loop,
            _create_async_runtime_engine(async_engine_class, async_engine_args),
        )
    except Exception:
        _shutdown_event_loop_thread(event_loop, event_loop_thread)
        raise
    return StoryVllmRuntime(
        llm=llm,
        max_model_len=int(runtime_kwargs["max_model_len"]),
        model_identifier=model_identifier,
        event_loop=event_loop,
        event_loop_thread=event_loop_thread,
    )


async def _create_async_runtime_engine(async_engine_class: Any, async_engine_args: Any) -> Any:
    return async_engine_class.from_engine_args(async_engine_args)


def close_generation_runtime(runtime: StoryVllmRuntime) -> None:
    if runtime.event_loop is None:
        return
    if runtime.event_loop.is_closed():
        return
    shutdown_background_loop = getattr(runtime.llm, "shutdown_background_loop", None)
    if callable(shutdown_background_loop):
        _run_in_event_loop(runtime.event_loop, _call_sync_function(shutdown_background_loop))
    _run_in_event_loop(runtime.event_loop, _drain_pending_event_loop_tasks())
    _shutdown_event_loop_thread(runtime.event_loop, runtime.event_loop_thread)


async def _stream_completion_text(
    runtime: StoryVllmRuntime,
    *,
    prompt: str,
    sampling_params: Any,
    text_chunk_callback: Any | None,
) -> str:
    streamed_text = ""
    async for request_output in runtime.llm.generate(prompt, sampling_params, request_id=uuid4().hex):
        completion_outputs = getattr(request_output, "outputs", None)
        if not isinstance(completion_outputs, list) or not completion_outputs:
            raise ValueError("vLLM generation output did not include any completion choices.")
        completion_text = getattr(completion_outputs[0], "text", None)
        if not isinstance(completion_text, str):
            raise TypeError("vLLM completion output did not contain text.")
        if not completion_text.startswith(streamed_text):
            raise ValueError("vLLM streaming output stopped being a monotonic text extension.")
        text_delta = completion_text[len(streamed_text) :]
        if text_delta and text_chunk_callback is not None:
            text_chunk_callback(text_delta)
        streamed_text = completion_text
    return streamed_text


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
    if runtime.event_loop is None:
        raise ValueError("StoryVllmRuntime.event_loop must be populated for pagestorm generation.")
    SamplingParams, StructuredOutputsParams = _load_sampling_params_types()
    sampling_params_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
    }
    if seed is not None:
        sampling_params_kwargs["seed"] = seed
    if top_k is not None:
        sampling_params_kwargs["top_k"] = top_k
    if structured_outputs is not None:
        sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs)
    sampling_params = SamplingParams(**sampling_params_kwargs)
    return _run_in_event_loop(
        runtime.event_loop,
        _stream_completion_text(
            runtime,
            prompt=prompt,
            sampling_params=sampling_params,
            text_chunk_callback=text_chunk_callback,
        ),
    )
