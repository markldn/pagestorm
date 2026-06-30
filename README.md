# PageStorm Studio — local GGUF harness

Runs the [PageStorm Research Preview 14B Full Book](https://huggingface.co/Pageshift-Entertainment/pagestorm-research-preview-14b-full-book)
staged story model through a `llama-server` GGUF backend, instead of the
upstream in-process vLLM path.

The upstream `pagestorm` package drives an in-process vLLM engine. Here, only
the generation backend is swapped: `pagestorm/vllm_runtime.py` is a drop-in
that talks to a `llama-server` `/completion` endpoint. All staging, parsing,
multi-chapter looping and validation are the upstream code, unchanged.

## Layout
- `pagestorm/` — upstream package **with `vllm_runtime.py` replaced** by the llama-server client.
- `serve.sh` — launches `llama-server` with the configured GGUF.
- `run.py` — drives staged full-book generation against the server.

## Model files
Set these paths for your machine:

- `PAGESTORM_BUNDLE_PATH` — local PageStorm bundle directory.
- `MODEL` — GGUF file passed to `llama-server`.
- `LLAMA_BIN` — `llama-server` executable, if it is not already on `PATH`.

## Usage
```bash
# 1. Start the server
MODEL=/path/to/pagestorm.gguf ./serve.sh

# 2. Generate a book (separate shell)
PAGESTORM_LLAMA_URL=http://<llama-server-host>:8091 \
PAGESTORM_BUNDLE_PATH=/path/to/pagestorm-bundle \
  python3 run.py \
    --prompt "A tense thriller set in Zurich's banking underworld" \
    --max-model-len 131072 \
    --output-directory out/zurich
```
Output: `out/<dir>/artifact.txt` (all stages) and `out/<dir>/<Book Title>.txt`.
Streamed text goes to stderr; a JSON run summary prints at the end.

## Web GUI — PageStorm Studio
```bash
./run_gui.sh
GUI_PORT=9000 ./run_gui.sh
GUI_HOST=<bind-host> ./run_gui.sh
```
`run_gui.sh` ensures the llama-server backend is up, then serves a single-page
studio (`gui/index.html`) via `gui_server.py` (Flask + SSE). It streams each
stage live (preview → story bible → chapters), shows a pipeline rail, extracts
the book title, marks per-stage regex validity, shows a live character counter
and labels planning stages as "planning…" (prose stages count words), and offers
copy/download of the finished manuscript. Generation runs server-side in a
worker thread and keeps going even if the browser tab closes; finished books
also land in `out/<slug>/`.

**Settings** (⚙, saved in localStorage):
- **LLM server URL** + Test button — passed per-run to the backend, so you can
  point at a llama-server on another machine.
- **Model name** (optional) — sent as `model` in the request (used by routers /
  llama-swap; ignored by plain llama.cpp). Auto-detected name shown as a hint.
- **Sampling** — temperature, top-k, top-p, min-p, repeat penalty. Blank = the
  orchestrator's tuned per-stage defaults; any value overrides globally.
- **Restore defaults** — resets all settings.
- **Theme** (Aurora / Noir / Paper / Parchment / Slate), **reading font**
  (Spectral / Literata / Inter / Georgia / Mono), **reading width** (Cozy /
  Default / Wide / Full).
- **Context** has an **Auto (detect)** option that reads the server's real
  `n_ctx` via `/api/server_info` (queries llama-server `/props`).

A **Stop** button (the Generate button becomes Stop while running) cancels a
generation mid-stream via `/api/stop` — it aborts at the next streamed token and
frees the GPU so you can start something else. Backed by a per-thread cancel
flag in the runtime (`GenerationCancelled`).

### serve.sh knobs (env vars)
- `PORT` (8091), `CTX` (131072), `KV_TYPE` (q8_0; set `f16` to disable KV quant),
  `GPU` (0), `NGL` (999), `MODEL`, `LLAMA_BIN`.

### Grammar-constrained decoding
vLLM constrained each stage to a regex. llama.cpp's `/completion` takes a GBNF
`grammar`, so `pagestorm/regex_to_gbnf.py` compiles the (regular) stage regexes
to GBNF and the runtime sends it automatically. The stage patterns are regular
(literals, char classes, alternation, bounded `{m,n}` repetition — no
backreferences), which maps cleanly onto GBNF.

- **On by default.** Set `PAGESTORM_GRAMMAR=0` to disable (best-effort decoding).
- If a pattern ever fails to convert, the runtime logs it and falls back to
  unconstrained for that stage rather than aborting.
- Verified: grammar-constrained output `re.fullmatch`es the original per-stage
  regex, including the large nested `book_plan` pattern.
- Two implementation details that matter:
  - **Named rules**: each regex group is emitted as its own GBNF rule, so nested
    `{m,n}` repetition unrolls into cheap rule *references* instead of inlining
    (and multiplying) the body. Without this, big patterns silently overflow
    llama.cpp's grammar and stop being enforced.
  - **`serve.sh` uses `--parallel 1`**: with the default 4 slots, llama.cpp
    splits the context (131072 / 4 = 32768 per request) AND grammar enforcement
    became intermittent across slots. One slot = full context + reliable grammar.
  - The compiler also handles the quirks of the *real* protocol regexes (which
    are richer than the model card's `story_stage_generation.py`): `\A`/`\Z`/`\z`
    anchors are dropped (not emitted as literal `A`/`Z`), and `\S` inside a
    negated class (`[^\S\n]` = horizontal whitespace) is resolved to an exact
    positive class. Verified `regex_valid=True` for book_preview and book_plan
    against `bundle.protocol`'s patterns.
- Trade-off: grammar-constrained decoding is slower, especially for large
  planning stages.

### Notes / limitations
- **`--max-model-len` must match the server's `CTX`** (default 131072).
- Large contexts require substantial memory. Tune `CTX`, `KV_TYPE`, `GPU`, and
  `NGL` for your hardware.
