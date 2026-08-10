"""Backend-agnostic device helpers.

PyTorch's ROCm build reuses the ``torch.cuda`` namespace, so a tensor placed on
``torch.device("cuda")`` lands on an AMD GPU and most CUDA-shaped code runs
unchanged. What does *not* survive the move is code that assumes a GPU is
present, that it is named "cuda" in user-facing text, or that ``torch.cuda``
exists at all when the model is being run on CPU.

Everything here is deliberately backend-neutral: the same call sites work on
NVIDIA (CUDA), AMD (ROCm/HIP), Apple silicon (MPS) and CPU.
"""

from typing import Union

import torch

# Accepted spellings that all mean "the default accelerator". ROCm exposes AMD
# GPUs through the CUDA API, so "rocm"/"hip" resolve to a cuda device.
_DEVICE_ALIASES = {
    "gpu": "cuda",
    "rocm": "cuda",
    "hip": "cuda",
}

_NO_GPU_HINT = (
    "No GPU is visible to PyTorch (torch.cuda.is_available() is False).\n"
    "  - On NVIDIA, install a CUDA build of torch.\n"
    "  - On AMD, install a ROCm build of torch matching your GPU architecture,\n"
    "    e.g. `pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch`\n"
    "    and check that `rocminfo` lists your GPU.\n"
    "  - Otherwise pass an explicit `--mula_device cpu --codec_device cpu`."
)


def is_rocm() -> bool:
    """True when torch was built against ROCm/HIP rather than CUDA."""
    return getattr(torch.version, "hip", None) is not None


def backend_name() -> str:
    """Human-readable name of the GPU backend torch was built with."""
    if is_rocm():
        return f"ROCm {torch.version.hip}"
    if torch.version.cuda is not None:
        return f"CUDA {torch.version.cuda}"
    return "CPU-only"


def resolve_device(spec: Union[str, torch.device, None]) -> torch.device:
    """Turn a user-supplied device string into a ``torch.device``.

    Beyond what ``torch.device`` accepts, this understands ``"auto"`` (pick the
    best available backend) and the aliases ``"gpu"``/``"rocm"``/``"hip"``, and
    fails with an actionable message instead of a late CUDA error when the
    requested accelerator is not usable.
    """
    if isinstance(spec, torch.device):
        device = spec
    else:
        text = "auto" if spec is None else str(spec).strip().lower()

        if text == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            kind, sep, index = text.partition(":")
            kind = _DEVICE_ALIASES.get(kind, kind)
            device = torch.device(f"{kind}{sep}{index}" if sep else kind)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}' but {_NO_GPU_HINT}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Requested device '{device}' but MPS is not available.")

    return device


def describe_device(device: torch.device) -> str:
    """``"cuda:0 (Radeon 8060S Graphics, ROCm 7.13...)"`` for logging."""
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)}, {backend_name()})"
    return str(device)


def memory_allocated(device: torch.device) -> int:
    """Bytes currently held by torch's allocator; 0 for backends without one."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def empty_cache(device: torch.device) -> None:
    """Return cached blocks to the driver, if the backend has a cache."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
