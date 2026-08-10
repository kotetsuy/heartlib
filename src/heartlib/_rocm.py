"""ROCm runtime defaults.

Applied at import time, before any convolution reaches MIOpen. Everything here
is a *default*: an environment variable the user already set always wins, and
none of it runs on a non-ROCm build.
"""

import os
from pathlib import Path

# Where scripts/tune_rocm.sh leaves its hipBLASLt tuning results.
TUNING_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "heartlib"


def tunableop_path(arch: str) -> Path:
    """Base filename for `arch`. PyTorch appends the device ordinal to it."""
    return TUNING_CACHE_DIR / f"tunableop_{arch}_.csv"


def _configure_tunableop(torch) -> None:
    """Replay hipBLASLt tuning results, if scripts/tune_rocm.sh produced any.

    TunableOp benchmarks the available hipBLASLt/rocBLAS kernels per GEMM shape
    and pins the winners; on gfx1151 that is worth ~40% on the generation loop.
    Tuning itself is expensive and gets *more* expensive with the prefix KV
    cache, because the attention GEMMs change shape every frame, so it is never
    done implicitly -- only replayed (PYTORCH_TUNABLEOP_TUNING=0). Shapes absent
    from the file quietly fall back to the default kernel.

    Cost is zero until someone runs the tuning script: without the cache
    directory this returns before touching the GPU.
    """
    if not TUNING_CACHE_DIR.is_dir() or not torch.cuda.is_available():
        return

    arch = torch.cuda.get_device_properties(0).gcnArchName
    base = tunableop_path(arch)
    # PyTorch inserts the device ordinal before the extension.
    if not base.with_name(f"{base.stem}0{base.suffix}").exists():
        return

    os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
    os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")
    os.environ.setdefault("PYTORCH_TUNABLEOP_FILENAME", str(base))


def configure_rocm_defaults() -> None:
    import torch

    if getattr(torch.version, "hip", None) is None:
        return

    # MIOpen's default "normal" find runs an exhaustive kernel search the first
    # time it sees each convolution shape. For HeartCodec's decoder on gfx1151
    # that costs ~2.5 minutes of every process's runtime -- and the solver it
    # settles on is an order of magnitude slower than the one the heuristic
    # search picks. Measured on gfx1151 (ScalarModel.decode, 2x128x744):
    #
    #     MIOPEN_FIND_MODE unset:  156.8 s first call, 18.2 s afterwards
    #     MIOPEN_FIND_MODE=FAST:     3.1 s first call,  1.9 s afterwards
    #
    # Outputs agree to ~1e-5 on a signal with RMS 0.57, i.e. floating-point
    # reassociation noise only. Set MIOPEN_FIND_MODE=NORMAL to opt back out.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

    _configure_tunableop(torch)
