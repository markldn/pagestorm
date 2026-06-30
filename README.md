# PageStorm Studio

PageStorm Studio is a local web and CLI harness for the PageStorm staged story
models. This fork keeps the PageStorm generation flow and adds a `llama-server`
backend path for running GGUF models locally.

The project is intended for people who want to run PageStorm outside the
upstream vLLM setup while keeping the same staged book-generation workflow:
preview, story planning, chapter planning, chapter writing, validation, and
artifact export.

## What Is Included

- A `llama-server` backed runtime for PageStorm generation.
- A browser-based studio for launching and reading generations.
- CLI helpers for starting a server and generating a book.
- Grammar support for PageStorm's staged output formats.
- Environment-based configuration for model paths, server URLs, and runtime
  settings.

## Repository Layout

- `pagestorm/` - PageStorm package code with the local server runtime.
- `gui/` - single-page browser interface.
- `gui_server.py` - Flask server for the studio.
- `run.py` - CLI entrypoint for book generation.
- `serve.sh` - helper for launching `llama-server`.
- `run.sh` and `run_gui.sh` - convenience wrappers.

## Basic Usage

Set paths for your local model files and server:

```bash
export MODEL=/path/to/pagestorm.gguf
export PAGESTORM_BUNDLE_PATH=/path/to/pagestorm-bundle
export PAGESTORM_LLAMA_URL=http://<llama-server-host>:8091
```

Start the model server:

```bash
./serve.sh
```

Generate from the CLI:

```bash
python3 run.py \
  --prompt "A tense thriller set in Zurich's banking underworld" \
  --max-model-len 131072 \
  --output-directory out/zurich
```

Or open the web studio:

```bash
./run_gui.sh
```

## Configuration

Common environment variables:

- `MODEL` - GGUF file used by `llama-server`.
- `LLAMA_BIN` - `llama-server` executable, if it is not on `PATH`.
- `PAGESTORM_BUNDLE_PATH` - local PageStorm bundle directory.
- `PAGESTORM_LLAMA_URL` - URL for the running model server.
- `PORT` - model server port.
- `GUI_HOST` and `GUI_PORT` - studio bind host and port.
- `CTX`, `KV_TYPE`, `GPU`, and `NGL` - llama-server runtime settings.

## Notes

This fork does not include model weights. Download the PageStorm model files
separately and point the environment variables at your local copies.

Generated books and local model files are ignored by git by default.
