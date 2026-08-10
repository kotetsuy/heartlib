#!/usr/bin/env bash
#
# Set up heartlib on an AMD GPU (ROCm) from a fresh clone.
#
#   ./scripts/setup_rocm.sh                     # venv + ROCm torch + heartlib + checkpoints
#   ./scripts/setup_rocm.sh --skip-checkpoints  # stop before the ~21 GB download
#
# Order matters: a ROCm torch must be installed *before* `pip install -e .`,
# otherwise pip resolves torch from PyPI and you get the CUDA build.
#
# Overridable:
#   GFX_ARCH       GPU architecture (default: detected via rocminfo)
#   TORCH_VERSION  default: 2.9.1+rocm7.13.0
#   TORCHAUDIO_VERSION  default: 2.9.0+rocm7.13.0
#   PYTHON_VERSION default: 3.12
#   VENV_DIR       default: .venv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_CHECKPOINTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-checkpoints) SKIP_CHECKPOINTS=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1+rocm7.13.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0+rocm7.13.0}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- GPU arch ---
log "Detecting GPU architecture"
if [[ -z "${GFX_ARCH:-}" ]]; then
    command -v rocminfo >/dev/null 2>&1 || die \
        "rocminfo not found. Install ROCm, or set GFX_ARCH=gfx1151 explicitly."
    GFX_ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v 'generic' | head -1)"
    [[ -n "$GFX_ARCH" ]] || die \
        "Could not detect a gfx architecture from rocminfo. Set GFX_ARCH manually."
fi
echo "    GPU architecture: $GFX_ARCH"

INDEX_URL="https://repo.amd.com/rocm/whl/${GFX_ARCH}/"
if ! curl -sfI "$INDEX_URL" >/dev/null 2>&1; then
    die "No AMD wheel index for ${GFX_ARCH} at ${INDEX_URL}
     Browse https://repo.amd.com/rocm/whl/ for available architectures, or use
     the generic index: TORCH_VERSION=... pip install --index-url \\
     https://download.pytorch.org/whl/rocm6.4 torch torchaudio"
fi
echo "    Wheel index:      $INDEX_URL"

# -------------------------------------------------------------- installer ---
if command -v uv >/dev/null 2>&1; then
    INSTALLER=uv
else
    INSTALLER=pip
    echo "    uv not found, falling back to python -m venv + pip"
fi

# ------------------------------------------------------------------- venv ---
# Re-running the script to refresh dependencies is a normal thing to do, so an
# existing virtualenv is reused rather than treated as an error.
if [[ -x "$VENV_DIR/bin/python" ]]; then
    log "Reusing existing virtualenv at ${VENV_DIR}"
    echo "    (delete it first if you want a clean rebuild)"
    VENV_EXISTS=1
else
    log "Creating virtualenv at ${VENV_DIR} (Python ${PYTHON_VERSION})"
    VENV_EXISTS=0
fi

if [[ "$INSTALLER" == uv ]]; then
    [[ "$VENV_EXISTS" == 1 ]] || uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    pip_install() { uv pip install --python "$VENV_DIR/bin/python" "$@"; }
else
    if [[ "$VENV_EXISTS" == 0 ]]; then
        command -v "python${PYTHON_VERSION}" >/dev/null 2>&1 || die \
            "python${PYTHON_VERSION} not found. Install it, or set PYTHON_VERSION."
        "python${PYTHON_VERSION}" -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    fi
    pip_install() { "$VENV_DIR/bin/python" -m pip install "$@"; }
fi
PYTHON="$VENV_DIR/bin/python"

# ------------------------------------------------------------- ROCm torch ---
log "Installing ROCm torch ${TORCH_VERSION} for ${GFX_ARCH}"
torch_install_args=(
    --index-url "$INDEX_URL"
    --extra-index-url https://pypi.org/simple
)
[[ "$INSTALLER" == uv ]] && torch_install_args+=(--index-strategy unsafe-best-match --prerelease allow)

if ! pip_install "${torch_install_args[@]}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" huggingface_hub; then
    echo "    Pinned versions unavailable for ${GFX_ARCH}; retrying unpinned."
    pip_install "${torch_install_args[@]}" torch torchaudio huggingface_hub
fi

log "Verifying the GPU is visible to torch"
"$PYTHON" - <<'EOF'
import sys
import torch

hip = getattr(torch.version, "hip", None)
print(f"    torch  {torch.__version__}")
print(f"    ROCm   {hip}")
if hip is None:
    sys.exit("error: this is not a ROCm build of torch.")
if not torch.cuda.is_available():
    sys.exit(
        "error: torch.cuda.is_available() is False.\n"
        "       Check `rocminfo` and that your user is in the 'render' and 'video' groups."
    )
print(f"    device {torch.cuda.get_device_name(0)}")
EOF

# --------------------------------------------------------------- heartlib ---
log "Installing heartlib (editable)"
pip_install -e .

# ------------------------------------------------------------ checkpoints ---
if [[ "$SKIP_CHECKPOINTS" == 1 ]]; then
    log "Skipping checkpoints (--skip-checkpoints)"
else
    log "Downloading checkpoints into ./ckpt (~21 GB, resumable)"
    "$VENV_DIR/bin/hf" download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
    "$VENV_DIR/bin/hf" download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
    "$VENV_DIR/bin/hf" download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'
fi

log "Done"
cat <<EOF
    Generate music with:

      ${VENV_DIR}/bin/python ./examples/run_music_generation.py \\
        --model_path=./ckpt --version="3B" --max_audio_length_ms=30000

    See docs/ROCM.md for gfx1151 notes and the optional TunableOp recipe.
EOF
