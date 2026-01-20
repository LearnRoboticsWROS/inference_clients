#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

$PYTHON -V

if [ ! -d ".venv-orch" ]; then
  $PYTHON -m venv .venv-orch
fi

source .venv-orch/bin/activate
python -m pip install --upgrade pip wheel setuptools

if [ -f requirements-orch.txt ]; then
  pip install -r requirements-orch.txt
else
  echo "requirements-orch.txt not found. Aborting."
  exit 1
fi

echo "Done (orchestrator). Python: $(python -V)"
pip --version
