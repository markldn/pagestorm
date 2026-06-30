#!/usr/bin/env bash
# Launch llama-server for a PageStorm GGUF.
#
# Uses the ROCm llama.cpp build whose runtime supports the `mistral3` arch.
# The harness talks to this server's /completion endpoint (raw prompts; the
# staged chat template is rendered client-side by pagestorm).
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-llama-server}"
MODEL="${MODEL:-}"
PORT="${PORT:-8091}"
HOST="${HOST:-${PAGESTORM_LLAMA_HOST:-localhost}}"
CTX="${CTX:-131072}"         # staged full-book generation benefits from a large context
NGL="${NGL:-999}"            # all layers on GPU
GPU="${GPU:-0}"
KV_TYPE="${KV_TYPE:-q8_0}"   # quantize KV cache to fit big context (set f16 to disable)

export HIP_VISIBLE_DEVICES="${GPU}"
export GGML_HIP_VISIBLE_DEVICES="${GPU}"

echo "[serve] model : ${MODEL}"
echo "[serve] ctx   : ${CTX}  ngl: ${NGL}  gpu: ${GPU}  host: ${HOST}  port: ${PORT}"

exec "${LLAMA_BIN}" \
  --model "${MODEL}" \
  --host "${HOST}" --port "${PORT}" \
  --ctx-size "${CTX}" \
  --parallel 1 \
  --n-gpu-layers "${NGL}" \
  --cache-type-k "${KV_TYPE}" \
  --cache-type-v "${KV_TYPE}" \
  -fa on \
  --no-webui
if [[ -z "${MODEL}" ]]; then
  echo "[serve] MODEL must point to the PageStorm GGUF file." >&2
  exit 2
fi
