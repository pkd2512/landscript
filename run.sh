#!/usr/bin/env bash
set -euo pipefail

REGION="${1:-bangalore}"
SCENES="${2:-1}"
COMPOSITE="${3:-true-color}"

# Cross-platform venv activation helper (macOS/Linux use venv/bin, Windows uses venv/Scripts)
activate_venv() {
  if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
  elif [ -f "venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source venv/Scripts/activate
  else
    echo "Error: could not find venv activation script (venv/bin/activate or venv/Scripts/activate)" >&2
    exit 1
  fi
}

case "${REGION}" in
  gallery|view|serve)
    activate_venv
    python gallery.py --region "${SCENES}" --open
    exit 0
    ;;
esac

echo "=== Landscript: $REGION ($SCENES scenes) ==="

# Pick a Python that satisfies the project requirement (>=3.10)
pick_python() {
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
      case "$version" in
        3.1[0-9]|3.[2-9][0-9]) echo "$candidate"; return 0 ;;
      esac
    fi
  done
  return 1
}

# Set up virtual env if needed
if [ ! -d "venv" ]; then
    PY=$(pick_python) || { echo "Error: need Python >=3.10 (try: brew install python@3.12)" >&2; exit 1; }
    echo "Creating virtual environment with $PY..."
    "$PY" -m venv venv
fi

activate_venv
# Ensure pip is new enough for PEP 660 editable installs from pyproject.toml
pip install -q --upgrade pip setuptools wheel
pip install -q -e .

echo ""
python run_pipeline.py --region "$REGION" --scenes "$SCENES" --composite "$COMPOSITE"

echo ""
echo "Gallery:  ./run.sh gallery $REGION"
echo "Retry with:  ./run.sh $REGION 1 false-color"
echo "Composites: true-color, false-color, swir, agriculture"