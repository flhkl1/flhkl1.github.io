#!/usr/bin/env bash
# Serve the portfolio locally, exactly the way GitHub Pages will.
#
#   ./tools/serve.sh            # http://localhost:8000/0/  (opens browser)
#   ./tools/serve.sh 3000       # pick a different port
#   ./tools/serve.sh --no-open  # don't launch a browser
#
# Serves from the REPO ROOT, not this folder, so that the "← Portfolio"
# link and every ../ relative path resolve the same as they do in production.
# Ctrl-C to stop.

set -euo pipefail

PORT=8000
OPEN=1
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN=0 ;;
    ''|*[!0-9]*) echo "unknown argument: $arg" >&2; exit 1 ;;
    *) PORT="$arg" ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
SLUG="$(basename "$PROJECT_DIR")"
URL="http://localhost:${PORT}/${SLUG}/"

# If the port is busy, walk forward until we find a free one.
while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  echo "port $PORT is in use, trying $((PORT+1))..."
  PORT=$((PORT+1))
  URL="http://localhost:${PORT}/${SLUG}/"
done

echo "root   $ROOT"
echo "page   $URL"
echo "stop   Ctrl-C"
echo

if [ "$OPEN" = "1" ]; then
  ( sleep 1; open "$URL" ) &
fi

cd "$ROOT"
# Range-aware server: plain http.server cannot seek media, so audio that is
# meant to start part-way into a track would begin at 0:00 locally.
exec python3 "$PROJECT_DIR/tools/serve.py" "$PORT"
