#!/usr/bin/env bash
set -euo pipefail

REGION="${1:-bangalore}"
SCENES="${2:-3}"

echo "=== Landscript: $REGION ($SCENES scenes) ==="

# Set up virtual env if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/Scripts/activate
pip install -q -e .

echo ""
python run_pipeline.py --region "$REGION" --scenes "$SCENES"
