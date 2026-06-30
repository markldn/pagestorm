#!/usr/bin/env bash
# Serve PageStorm Studio (web GUI). Ensures the llama-server backend is running,
# then launches the Flask UI server.
#
#   ./run_gui.sh
#   GUI_PORT=9000 ./run_gui.sh
#   GUI_HOST=<bind-host> ./run_gui.sh
#   START_SERVER=0 ./run_gui.sh     # don't auto-start llama-server
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8091}"                       # llama-server port
PAGESTORM_LLAMA_HOST="${PAGESTORM_LLAMA_HOST:-localhost}"
URL="${PAGESTORM_LLAMA_URL:-http://${PAGESTORM_LLAMA_HOST}:${PORT}}"
GUI_PORT="${GUI_PORT:-8092}"
GUI_HOST="${GUI_HOST:-localhost}"
START_SERVER="${START_SERVER:-1}"

export PAGESTORM_LLAMA_URL="${URL}"

# Bring up the model server if it isn't answering.
if [[ "${START_SERVER}" == "1" ]] && ! curl -s "${URL}/health" 2>/dev/null | grep -q '"ok"'; then
  echo "[gui] starting llama-server backend (${URL}) ..."
  PORT="${PORT}" setsid bash -c "'${HERE}/serve.sh' > '${HERE}/serve.log' 2>&1" < /dev/null &
  disown || true
  echo -n "[gui] waiting for model"
  until curl -s "${URL}/health" 2>/dev/null | grep -q '"ok"'; do echo -n "."; sleep 2; done
  echo " ready."
fi

echo "[gui] PageStorm Studio  ->  http://${GUI_HOST}:${GUI_PORT}"
exec env GUI_PORT="${GUI_PORT}" GUI_HOST="${GUI_HOST}" python3 "${HERE}/gui_server.py"
