#!/usr/bin/env bash
#
# Tune hipBLASLt GEMM kernels for this GPU, once.
#
#   ./scripts/tune_rocm.sh [--model_path ./ckpt]
#
# Runs one short generation with PyTorch's TunableOp in tuning mode, which
# benchmarks the available hipBLASLt/rocBLAS kernels for every GEMM shape the
# model uses and records the winners. heartlib picks the results up
# automatically on later runs (see src/heartlib/_rocm.py) and replays them
# without ever tuning again.
#
# Measured on gfx1151, 30 s of audio: 9.53 -> 13.67 it/s (RTF 1.05 -> 0.91).
# For the 4-minute default, 7.78 -> 9.47 it/s (RTF 1.61 -> 1.32) using the same
# file -- most of the win comes from the fixed-shape projections, so a short
# tuning run transfers to long generations.
#
# This run is itself slow (~1/3 of normal speed) because tuning happens inline.
# It is a one-off; delete the cache file to redo it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="./ckpt"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
[[ -x "$PYTHON" ]] || { echo "error: $PYTHON not found. Run ./scripts/setup_rocm.sh first." >&2; exit 1; }

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

log "Locating the tuning cache"
PATHS="$("$PYTHON" - <<'EOF'
import sys
import torch
from heartlib._rocm import TUNING_CACHE_DIR, tunableop_path

if getattr(torch.version, "hip", None) is None:
    sys.exit("this is not a ROCm build of torch; TunableOp tuning is ROCm-only.")
if not torch.cuda.is_available():
    sys.exit("no GPU visible to torch.")

TUNING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
base = tunableop_path(torch.cuda.get_device_properties(0).gcnArchName)
print(base, base.with_name(f"{base.stem}0{base.suffix}"))
EOF
)" || { echo "error: could not determine the tuning cache path" >&2; exit 1; }
read -r TUNE_BASE TUNE_FILE <<<"$PATHS"
echo "    Results will be written to: $TUNE_FILE"

if [[ -f "$TUNE_FILE" ]]; then
    log "Already tuned"
    echo "    $TUNE_FILE exists; delete it to re-tune."
    exit 0
fi

log "Tuning (one short generation, deliberately slow)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PYTORCH_TUNABLEOP_ENABLED=1 \
PYTORCH_TUNABLEOP_TUNING=1 \
PYTORCH_TUNABLEOP_FILENAME="$TUNE_BASE" \
    "$PYTHON" ./examples/run_music_generation.py \
        --model_path="$MODEL_PATH" --version="3B" \
        --max_audio_length_ms=20000 \
        --save_path="$SCRATCH/tuning.mp3"

log "Done"
echo "    $(wc -l < "$TUNE_FILE") entries in $TUNE_FILE"
echo "    Later runs replay these automatically; nothing else to configure."
