#!/usr/bin/env bash
# Serve the playground over HTTP so the browser can fetch the shader files.
# Opening index.html directly (file://) won't work — fetch() is blocked there.
set -euo pipefail
PORT="${1:-8080}"
cd "$(dirname "$0")"
echo "warp shader playground → http://localhost:${PORT}/"
exec python3 -m http.server "${PORT}"
