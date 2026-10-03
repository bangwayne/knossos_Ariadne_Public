#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT=knossos_natural_process_v1/assets/prompts/natural_node_asset_pilot_24.jsonl \
OUT_DIR=knossos_natural_process_v1/assets/pilot_raw \
CONCURRENCY="${CONCURRENCY:-4}" \
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}" \
bash knossos_natural_process_v1/scripts/run_natural_icon_full.sh "$@"
