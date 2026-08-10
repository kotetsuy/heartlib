# Running heartlib on AMD GPUs (ROCm)

heartlib was written for NVIDIA/CUDA. This branch makes it run on AMD GPUs
through ROCm. Nothing here is AMD-only: the same code still runs on CUDA, on
Apple silicon (MPS) and on CPU.

Verified on:

| | |
|---|---|
| GPU | Radeon 8060S (Ryzen AI MAX+ 395, **gfx1151**, 48 GB VRAM) |
| OS | Ubuntu 26.04 |
| torch | 2.9.1+rocm7.13.0 (Python 3.12) |
| Model | HeartMuLa-oss-3B-happy-new-year + HeartCodec-oss-20260123 |

## Why CUDA code mostly "just works" on ROCm

PyTorch's ROCm build reuses the `torch.cuda` namespace: `torch.device("cuda")`,
`torch.cuda.is_available()` and `torch.autocast(device_type="cuda")` all address
the AMD GPU. `torch.version.cuda` is `None` and `torch.version.hip` is set — that
is the reliable way to tell the two apart.

So the porting work is not about rewriting kernels. It is about (a) the
dependency set, which contains CUDA-only wheels, and (b) a handful of places
that hard-code `cuda`.

## Install

On a fresh clone, one command does everything — detect the GPU architecture,
build the virtualenv, install a ROCm torch and heartlib, and fetch the
checkpoints:

```bash
./scripts/setup_rocm.sh                     # ~21 GB of checkpoints included
./scripts/setup_rocm.sh --skip-checkpoints  # code and environment only
```

It is safe to re-run: an existing `.venv` is reused rather than rebuilt.
`GFX_ARCH`, `TORCH_VERSION`, `PYTHON_VERSION` and `VENV_DIR` override the
defaults. The rest of this section is what the script does, by hand.

**Do not `pip install -e .` first** — that would pull the CUDA build of torch
from PyPI. Install a ROCm torch matching your GPU architecture, then heartlib.

```bash
# Pick the index for your architecture. gfx1151 = Strix Halo / Radeon 8060S.
# Discover yours with: rocminfo | grep gfx
uv venv --python 3.12
uv pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease allow \
  torch==2.9.1+rocm7.13.0 torchaudio==2.9.0+rocm7.13.0

uv pip install -e .
```

For other architectures (gfx90a, gfx942, gfx1100, …) use the matching
`https://repo.amd.com/rocm/whl/<arch>/` index, or the generic
`https://download.pytorch.org/whl/rocm6.x` index.

Check the install:

```bash
python -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 7.13.99004-...  True  Radeon 8060S Graphics
```

Then download checkpoints and run exactly as the main README describes:

```bash
python ./examples/run_music_generation.py --model_path=./ckpt --version="3B"
```

`--mula_device` / `--codec_device` now default to `auto`, and accept `auto`,
`cuda`, `cuda:0`, `rocm`, `gpu`, `mps` and `cpu`.

## What had to change, and why

### 1. `bitsandbytes` is no longer a hard dependency

`bitsandbytes` has no gfx1151 build; on ROCm it either fails to load its native
library or silently falls back to CPU. Nothing under `src/heartlib` imports it,
so it moved to the `quant` extra (`pip install -e ".[quant]"`). `torchvision`,
`einops`, `ipykernel`, `traitlets` and `traittypes` were also unused and were
dropped or made extras; `modelscope` (a checkpoint-download convenience) moved
to the `download` extra.

`torchao` **stayed** a hard dependency — `import torchtune` hard-imports it —
but the generic PyPI wheel imports fine against a ROCm torch, because its CUDA
extension is optional.

### 2. `torch.cuda.*` calls became backend-agnostic

`src/heartlib/_device.py` wraps device selection (`auto`/`rocm`/`gpu` aliases,
an actionable error when no GPU is visible) and the allocator helpers
(`memory_allocated`, `empty_cache`) that `--lazy_load` used to call as
`torch.cuda.*` — which crashes on a CPU-only or MPS run.

### 3. `torchaudio.save` no longer works — soundfile fallback

From torchaudio 2.9, `torchaudio.save` delegates to TorchCodec, which is not
part of the ROCm wheel set:

```
ImportError: TorchCodec is required for save_with_torchcodec.
```

`src/heartlib/_audio.py` tries `torchaudio.save` first and falls back to
soundfile (already a dependency; libsndfile encodes mp3, flac and wav). This is
a torchaudio-version problem rather than an AMD one — CUDA users on torchaudio
2.9 hit it too.

### 4. MIOpen kernel-search default

`heartlib/__init__.py` sets `MIOPEN_FIND_MODE=FAST` on ROCm builds — a 2.5x
end-to-end win. See [the gfx1151 section](#miopen-exhaustive-search--miopen_find_modefast-is-set-for-you).

## gfx1151 specifics

### Attention runs on the math backend

Both fused SDPA kernels are runtime-disabled on this GPU:

```
Flash attention kernel not used because: Flash attention has been runtime disabled.
Memory efficient kernel not used because: Mem Efficient attention on Current AMD GPU
  is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
```

Setting `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` does **not** help on gfx1151
— AOTriton has no kernel for this architecture and the call then hard-fails with
`No available kernel. Aborting execution.` PyTorch's automatic fallback to the
math backend is correct here and is what you want; leave the variable unset.

This is fine for correctness and cheap for this workload: decoding runs with a
KV cache at sequence length 1, so no large attention matrix is ever
materialised. It does mean flash-attention speedups are simply unavailable.

The warning is emitted once per call site and is harmless.

### MIOpen exhaustive search — `MIOPEN_FIND_MODE=FAST` is set for you

This was the single biggest problem, and it is not visible as an error. gfx1151
ships no pre-tuned MIOpen database:

```
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: ".../gfx1151_20.HIP.fdb.txt"
MIOpen(HIP): Warning [IsEnoughWorkspace] [EvaluateInvokers] Solver <GemmFwdRest> ...
```

so MIOpen falls back to an exhaustive kernel search for HeartCodec's decoder
convolutions — every process, on every run, because the results it caches are
not re-used. Worse, the solver it settles on (`GemmFwdRest`, running without
workspace) is an order of magnitude slower than the one the heuristic search
picks. Measured on gfx1151, `ScalarModel.decode` with a 2x128x744 latent:

| `MIOPEN_FIND_MODE` | first call | steady state |
|---|---|---|
| unset (normal) | 156.8 s | 18.2 s |
| `FAST` | 3.1 s | 1.9 s |

Outputs agree to within 1.5e-5 on a signal of RMS 0.57 — floating-point
reassociation noise, nothing more.

`heartlib/__init__.py` therefore sets `MIOPEN_FIND_MODE=FAST` as a *default* on
ROCm builds (see `src/heartlib/_rocm.py`); an explicitly set value always wins,
so `MIOPEN_FIND_MODE=NORMAL` opts back out. End-to-end, on 20 s of audio:

| | wall clock |
|---|---|
| first ever run (cold MIOpen + cold page cache) | 7 m 23 s |
| warm run, MIOpen default | 4 m 18 s |
| warm run, `MIOPEN_FIND_MODE=FAST` | **1 m 44 s** |

### Optional: hipBLASLt TunableOp, ~19% on the generation loop

PyTorch's TunableOp benchmarks the available hipBLASLt/rocBLAS GEMM kernels for
your exact shapes and pins the winners. Tune once:

```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$PWD/tunable.csv \
  python ./examples/run_music_generation.py --model_path=./ckpt --max_audio_length_ms=10000
```

That writes `tunable0.csv` (the device ordinal is appended). Re-use it, with
tuning off, on every later run:

```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
PYTORCH_TUNABLEOP_FILENAME=$PWD/tunable.csv \
  python ./examples/run_music_generation.py --model_path=./ckpt
```

Measured on the HeartMuLa decode loop: 3.42 it/s baseline, 2.97 it/s during the
tuning run, **4.08 it/s** once tuned. It is opt-in rather than a default
because it writes a machine-specific file into your working directory.

### Performance summary

On gfx1151, HeartMuLa's decode loop runs at ~3.4 frames/s (~4.1 tuned) against
the 12.5 Hz frame rate, i.e. RTF ≈ 3.7 (≈ 3.1 tuned) — slower than the RTF ≈ 1.0
the README quotes for datacenter NVIDIA parts. The loop is dominated by the 3B
backbone, one forward per 80 ms frame, on an APU whose weights live in
LPDDR5X. Nothing about it is broken; it is bandwidth.

### `xnack 'Off' was requested for a processor that does not support it`

Cosmetic. gfx1151 does not implement XNACK; the code object requests it off
anyway. Nothing to do.

## Setting this up on another machine

Only source is in git — the virtualenv and the ~21 GB of checkpoints are not,
and are both re-created by the setup script.

```bash
git clone -b rocm-support git@github.com:kotetsuy/heartlib.git
cd heartlib
./scripts/setup_rocm.sh
```

If the other machine has a different AMD GPU, the script picks the matching
wheel index from `rocminfo` on its own; only the gfx1151-specific performance
notes below are architecture-bound. To pull in later upstream work:

```bash
git remote add upstream git@github.com:HeartMuLa/heartlib.git   # once
git fetch upstream && git rebase upstream/main
```

## Troubleshooting

**`hipErrorInvalidImage` / `kpack_load_code_object failed with error: 13`** —
you installed a multi-architecture wheel rather than a gfx1151 one. Use the
`whl/gfx1151/` index.

**`torch.cuda.is_available()` is False** — check `rocminfo` lists your GPU and
that your user is in the `render` and `video` groups.

**Out of memory** — on an APU, VRAM is carved out of system RAM. Use
`--lazy_load true` so HeartMuLa is freed before HeartCodec loads, and lower
`--max_audio_length_ms`.

**Do not set `HSA_OVERRIDE_GFX_VERSION`** when using a native gfx1151 build.
Overriding the architecture breaks it.
