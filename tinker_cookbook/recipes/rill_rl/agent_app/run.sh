#!/usr/bin/env bash
# Launch the standalone RILL agent app (UI + backend) on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY first (or pass a Base URL in the UI)." >&2
fi

# Run from the repo root so the package import path resolves.
cd ../../../..
exec python -m tinker_cookbook.recipes.rill_rl.agent_app.server
