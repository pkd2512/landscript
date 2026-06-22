#!/usr/bin/env bash
#
# Landscript convenience runner.
#
# Subcommands:
#   ./run.sh setup                   create venv, install deps
#   ./run.sh region   <id>           run pipeline for one region
#                                    (e.g. in-kutch-rann, bangalore, …)
#   ./run.sh country  <id> [scenes]  run pipeline for a whole country
#                                    (e.g. india)
#   ./run.sh regen    [region]       rewrite glyph PNGs from cached source
#                                    tiles (no STAC / no network)
#   ./run.sh gallery  [region]       serve the gallery + open browser
#
# Defaults: scenes=1, composite=true-color, region/country=india.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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

activate_venv() {
  if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
  elif [ -f "venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source venv/Scripts/activate
  else
    echo "Error: venv not found. Run \`./run.sh setup\` first." >&2
    exit 1
  fi
}

ensure_venv() {
  if [ ! -d "venv" ]; then
    PY=$(pick_python) || {
      echo "Error: need Python >= 3.10 (try: brew install python@3.12)" >&2
      exit 1
    }
    echo "Creating virtual environment with $PY ..."
    "$PY" -m venv venv
  fi
  activate_venv
  pip install -q --upgrade pip setuptools wheel
  pip install -q -e .
}

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^#$//;s/^# \{0,1\}//'
  exit 1
}

# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

cmd_setup() {
  ensure_venv
  echo "Setup complete. Try:  ./run.sh region in-kutch-rann"
}

cmd_region() {
  local region="${1:-bangalore}"
  local scenes="${2:-1}"
  local composite="${3:-true-color}"
  ensure_venv
  echo "=== Landscript region: $region ($scenes scene[s], $composite) ==="
  python run_pipeline.py --region "$region" --scenes "$scenes" --composite "$composite"
  echo
  echo "Next:  ./run.sh gallery $region"
}

cmd_country() {
  local country="${1:-india}"
  local scenes="${2:-1}"
  local top="${3:-200}"
  ensure_venv
  echo "=== Landscript country: $country ($scenes scene[s] per region, top $top) ==="
  python run_pipeline.py --country "$country" --scenes "$scenes" --top "$top"
  echo
  echo "Next:  ./run.sh gallery $country"
}

cmd_regen() {
  local region="${1:-india}"
  ensure_venv
  echo "=== Rebuilding glyph PNGs from cached tiles for: $region ==="
  python regen_pngs.py --region "$region"
}

cmd_gallery() {
  local region="${1:-india}"
  local port="${2:-8080}"
  ensure_venv
  python gallery.py --region "$region" --port "$port" --open
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

# Back-compat: `./run.sh <region>` (legacy positional) still works.
case "${1:-}" in
  ""|-h|--help|help) usage ;;

  setup)              shift; cmd_setup "$@" ;;
  region)             shift; cmd_region "$@" ;;
  country)            shift; cmd_country "$@" ;;
  regen)              shift; cmd_regen "$@" ;;
  gallery|view|serve) shift; cmd_gallery "$@" ;;

  *)
    # Legacy: treat as a region name + optional scenes + composite.
    cmd_region "$1" "${2:-1}" "${3:-true-color}"
    ;;
esac