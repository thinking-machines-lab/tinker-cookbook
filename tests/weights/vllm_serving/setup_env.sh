#!/usr/bin/env bash
# Create an isolated venv for vLLM adapter serving tests.
#
# Usage:
#   bash tests/weights/vllm_serving/setup_env.sh
#   /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/ -v -s
#
set -euo pipefail

VENV_DIR="${VLLM_TEST_ENV:-/tmp/vllm-test-env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing vLLM and test dependencies ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "Installing tinker-cookbook (editable) ..."
"$VENV_DIR/bin/pip" install --quiet -e "$REPO_ROOT"

echo ""
echo "Done. Run tests with:"
echo "  $VENV_DIR/bin/python -m pytest tests/weights/vllm_serving/ -v -s"
