#!/usr/bin/env python3
"""PageStorm Studio — a slick web UI for staged book generation.

Serves a single-page UI and streams the orchestrator's stage events over SSE:
  stage_start  -> a new stage began (role)
  chunk        -> streamed text delta for the current stage
  stage_complete -> a stage finished (role, full text, regex validity)
  done / error -> terminal events

Generation runs in a worker thread; callbacks push events onto a queue that the
SSE response drains. Talks to llama-server via PAGESTORM_LLAMA_URL (see serve.sh).
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flask import Flask, Response, jsonify, request, send_file

from pagestorm import vllm_runtime
from pagestorm.artifacts import write_generation_outputs
from pagestorm.bundle import load_story_bundle
from pagestorm.orchestrator import generate_full_book

DEFAULT_BUNDLE_PATH = os.environ.get(
    "PAGESTORM_BUNDLE_PATH",
    str(HERE / "models" / "pagestorm-research-preview-14b-full-book"),
)
DEFAULT_LLAMA_HOST = os.environ.get("PAGESTORM_LLAMA_HOST", "localhost")
DEFAULT_LLAMA_PORT = os.environ.get("PAGESTORM_LLAMA_PORT", "8091")
DEFAULT_LLAMA_URL = os.environ.get("PAGESTORM_LLAMA_URL", f"http://{DEFAULT_LLAMA_HOST}:{DEFAULT_LLAMA_PORT}")

app = Flask(__name__)

# Active generations: job_id -> cancel Event (set by /api/stop).
JOBS: "dict[str, object]" = {}


def _configured_server_url() -> str:
    return (request.args.get("server") or DEFAULT_LLAMA_URL).rstrip("/")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48] or "book"


@app.route("/")
def index():
    return send_file(HERE / "gui" / "index.html")


@app.route("/api/health")
def health():
    import requests as _rq

    url = _configured_server_url()
    try:
        ok = _rq.get(f"{url}/health", timeout=4).json().get("status") == "ok"
    except Exception:
        ok = False
    return jsonify({"server": url, "ready": ok})


@app.route("/api/server_info")
def server_info():
    """Autodetect what the llama-server is serving: context size + model name."""
    import requests as _rq

    url = _configured_server_url()
    info = {"server": url, "ready": False, "n_ctx": None, "model": None}
    try:
        props = _rq.get(f"{url}/props", timeout=4).json()
        info["ready"] = True
        gen = props.get("default_generation_settings") or {}
        info["n_ctx"] = gen.get("n_ctx") or props.get("n_ctx")
        model_path = props.get("model_path") or props.get("model") or gen.get("model")
        if isinstance(model_path, str):
            info["model"] = os.path.basename(model_path)
    except Exception:
        pass
    return jsonify(info)


@app.route("/api/models")
def models():
    """List available model IDs from llama-server or an OpenAI-compatible router."""
    import requests as _rq

    url = _configured_server_url()
    result = {"server": url, "ready": False, "models": [], "source": None}

    def add_model(items: list[str], value, *, path_name: bool = False) -> None:
        if isinstance(value, str):
            name = os.path.basename(value.strip()) if path_name else value.strip()
            if name and name not in items:
                items.append(name)

    try:
        response = _rq.get(f"{url}/v1/models", timeout=6)
        if response.ok:
            data = response.json()
            models: list[str] = []
            for item in data.get("data", []):
                if isinstance(item, dict):
                    add_model(models, item.get("id") or item.get("name") or item.get("model"))
                else:
                    add_model(models, item)
            if models:
                result.update({"ready": True, "models": models, "source": "/v1/models"})
                return jsonify(result)
    except Exception:
        pass

    try:
        response = _rq.get(f"{url}/models", timeout=6)
        if response.ok:
            data = response.json()
            raw_models = data.get("models") if isinstance(data, dict) else data
            models = []
            if isinstance(raw_models, list):
                for item in raw_models:
                    if isinstance(item, dict):
                        add_model(models, item.get("id") or item.get("name") or item.get("model"))
                    else:
                        add_model(models, item)
            if models:
                result.update({"ready": True, "models": models, "source": "/models"})
                return jsonify(result)
    except Exception:
        pass

    try:
        props = _rq.get(f"{url}/props", timeout=6).json()
        models = []
        gen = props.get("default_generation_settings") or {}
        model_path = props.get("model_path")
        if model_path:
            add_model(models, model_path, path_name=True)
        else:
            add_model(models, props.get("model") or gen.get("model"))
        if models:
            result.update({"ready": True, "models": models, "source": "/props"})
    except Exception:
        pass
    return jsonify(result)


@app.route("/api/stop")
def api_stop():
    job_id = (request.args.get("job") or "").strip()
    ev = JOBS.get(job_id)
    if ev is not None:
        ev.set()
        return jsonify({"stopped": True, "job": job_id})
    return jsonify({"stopped": False, "job": job_id})


@app.route("/api/generate")
def api_generate():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    max_model_len = int(request.args.get("max_model_len", "131072"))
    chapter_limit_raw = request.args.get("chapter_limit", "").strip()
    later_chapter_limit = int(chapter_limit_raw) if chapter_limit_raw.isdigit() else None
    server_override = (request.args.get("server") or "").strip()
    model_override = (request.args.get("model") or "").strip()
    sampling = {
        "PAGESTORM_TEMPERATURE": (request.args.get("temperature") or "").strip(),
        "PAGESTORM_TOP_K": (request.args.get("top_k") or "").strip(),
        "PAGESTORM_TOP_P": (request.args.get("top_p") or "").strip(),
        "PAGESTORM_MIN_P": (request.args.get("min_p") or "").strip(),
        "PAGESTORM_REPEAT_PENALTY": (request.args.get("repeat_penalty") or "").strip(),
    }
    job_id = (request.args.get("job") or "").strip()
    cancel_event = threading.Event()
    if job_id:
        JOBS[job_id] = cancel_event

    events: "queue.Queue[dict]" = queue.Queue()
    current = {"role": None}

    def on_start(role: str) -> None:
        current["role"] = role
        events.put({"type": "stage_start", "role": role, "t": time.time()})

    def on_chunk(text_delta: str) -> None:
        events.put({"type": "chunk", "role": current["role"], "text": text_delta})

    def on_complete(stage) -> None:
        events.put({
            "type": "stage_complete",
            "role": stage.role,
            "text": stage.text,
            "regex_valid": stage.regex_valid,
            "validation_error": stage.validation_error,
        })

    def worker() -> None:
        vllm_runtime.register_cancel(cancel_event)
        try:
            if server_override:
                os.environ["PAGESTORM_LLAMA_URL"] = server_override
            os.environ["PAGESTORM_MODEL"] = model_override
            for env_key, val in sampling.items():
                os.environ[env_key] = val
            bundle = load_story_bundle(profile_name="14b", bundle_path=DEFAULT_BUNDLE_PATH)
            run = generate_full_book(
                bundle,
                prompt=prompt,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.0,
                max_model_len=max_model_len,
                later_chapter_limit=later_chapter_limit,
                stage_start_callback=on_start,
                stage_text_chunk_callback=on_chunk,
                stage_output_callback=on_complete,
            )
            outdir = HERE / "out" / _slug(prompt)
            outdir.mkdir(parents=True, exist_ok=True)
            write_generation_outputs(str(outdir), run)
            events.put({
                "type": "done",
                "validation_success": run.validation_success,
                "model": run.model,
                "later_chapter_count": run.later_chapter_count,
                "output_directory": str(outdir),
            })
        except vllm_runtime.GenerationCancelled:
            events.put({"type": "stopped"})
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            vllm_runtime.unregister_cancel()
            JOBS.pop(job_id, None)
            events.put({"type": "__end__"})

    def stream():
        threading.Thread(target=worker, daemon=True).start()
        yield 'data: {"type":"started"}\n\n'
        while True:
            try:
                ev = events.get(timeout=10)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if ev.get("type") == "__end__":
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("GUI_PORT", "8092"))
    host = os.environ.get("GUI_HOST", "localhost")
    print(f"[gui] PageStorm Studio on http://{host}:{port}", flush=True)
    app.run(host=host, port=port, threaded=True)
