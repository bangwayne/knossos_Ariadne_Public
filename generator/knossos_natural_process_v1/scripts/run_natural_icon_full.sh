#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
IMAGE_GEN="${IMAGE_GEN:-${HOME}/.codex/skills/.system/imagegen/scripts/image_gen.py}"
CONFIG="${CONFIG:-openai_image_config.yaml}"
INPUT="${INPUT:-knossos_natural_process_v1/assets/prompts/natural_node_asset_jobs.jsonl}"
OUT_DIR="${OUT_DIR:-knossos_natural_process_v1/assets/renderer_asset_pool}"
MODEL="${MODEL:-gpt-image-1-mini}"
SIZE="${SIZE:-1024x1024}"
QUALITY="${QUALITY:-medium}"
BACKGROUND="${BACKGROUND:-opaque}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-png}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

eval "$("${PYTHON}" -c '
from pathlib import Path
import shlex
import sys

config = Path(sys.argv[1])
text = config.read_text(encoding="utf-8").splitlines()
current = None
vals = {}
for line in text:
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    indent = len(line) - len(line.lstrip())
    if ":" not in line:
        continue
    key, raw = line.split(":", 1)
    key = key.strip()
    val = raw.strip().strip("\"").strip(chr(39))
    if indent == 0:
        current = key
    elif current:
        vals[f"{current}.{key}"] = val

api_key = vals.get("api.key", "")
api_base = vals.get("api.base", "")
if not api_key:
    raise SystemExit(f"api.key missing from {config}")
print("export OPENAI_API_KEY=" + shlex.quote(api_key))
if api_base:
    print("export OPENAI_BASE_URL=" + shlex.quote(api_base))
' "${CONFIG}")"

mkdir -p "${OUT_DIR}"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN_FLAG="--dry-run"
fi

PYTHONPATH="./vendor${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" "${IMAGE_GEN}" generate-batch \
  --input "${INPUT}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL}" \
  --background "${BACKGROUND}" \
  --size "${SIZE}" \
  --quality "${QUALITY}" \
  --output-format "${OUTPUT_FORMAT}" \
  --concurrency "${CONCURRENCY}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --force \
  ${DRY_RUN_FLAG}
