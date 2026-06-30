#!/usr/bin/env bash
# Convenience wrapper: ensure the llama-server is up, then run staged generation.
#
# Usage:
#   ./run.sh "A tense thriller in Zurich"            # auto output dir under out/
#   ./run.sh "prompt..." out/mybook                  # explicit output dir
#   ./run.sh --validate "prompt..."                  # strict validation pass
#
# Honors the same env knobs as serve.sh (PORT, etc). Set START_SERVER=0 to skip
# the auto-start (e.g. server already running elsewhere).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8091}"
PAGESTORM_LLAMA_HOST="${PAGESTORM_LLAMA_HOST:-localhost}"
URL="${PAGESTORM_LLAMA_URL:-http://${PAGESTORM_LLAMA_HOST}:${PORT}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
START_SERVER="${START_SERVER:-1}"

VALIDATE_FLAG=""
if [[ "${1:-}" == "--validate" ]]; then
  VALIDATE_FLAG="--validate"
  shift
fi

PROMPT="${1:-}"
if [[ -z "${PROMPT}" ]]; then
  echo "usage: $0 [--validate] \"<prompt>\" [output_dir]" >&2
  exit 2
fi
# Default output dir: out/<slugified-prompt>
OUTDIR="${2:-}"
if [[ -z "${OUTDIR}" ]]; then
  SLUG="$(echo "${PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | cut -c1-40 | sed 's/-$//')"
  OUTDIR="${HERE}/out/${SLUG:-book}"
fi
mkdir -p "${OUTDIR}"

# Start the server if it isn't answering yet.
if [[ "${START_SERVER}" == "1" ]] && ! curl -s "${URL}/health" 2>/dev/null | grep -q '"ok"'; then
  echo "[run] starting llama-server (${URL}) ..."
  PORT="${PORT}" nohup "${HERE}/serve.sh" > "${HERE}/serve.log" 2>&1 &
  until curl -s "${URL}/health" 2>/dev/null | grep -q '"ok"'; do sleep 2; done
  echo "[run] server healthy."
fi

echo "[run] prompt   : ${PROMPT}"
echo "[run] output   : ${OUTDIR}"
echo "[run] ctx      : ${MAX_MODEL_LEN}  url: ${URL}"

PAGESTORM_LLAMA_URL="${URL}" exec python3 "${HERE}/run.py" \
  --prompt "${PROMPT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --output-directory "${OUTDIR}" \
  ${VALIDATE_FLAG}
