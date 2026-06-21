#!/usr/bin/env bash
set -euo pipefail

REGION="${1:-bangalore}"
SCENES="${2:-1}"
COMPOSITE="${3:-true-color}"

case "${REGION}" in
  gallery|view|serve)
    source venv/Scripts/activate
    python gallery.py --region "${SCENES}" --open
    exit 0
    ;;
esac

echo "=== Landscript: $REGION ($SCENES scenes) ==="

# Set up virtual env if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/Scripts/activate
pip install -q -e .

echo ""
python run_pipeline.py --region "$REGION" --scenes "$SCENES" --composite "$COMPOSITE"

echo ""
echo "Gallery:  ./run.sh gallery $REGION"
echo "Retry with:  ./run.sh $REGION 1 false-color"
echo "Composites: true-color, false-color, swir, agriculture"
