#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

# prereq check
$PYTHON -V

if [ ! -d ".venv" ]; then
  $PYTHON -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Aborting."
  exit 1
fi

echo "Done. Python: $(python -V)"
pip --version