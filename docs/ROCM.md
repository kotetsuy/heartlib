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

Then, once per machine, tune the GEMM kernels — worth ~35% on the generation
loop and picked up automatically afterwards:

```bash
./scripts/tune_rocm.sh
```

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

### hipBLASLt TunableOp — run `scripts/tune_rocm.sh` once, get ~35%

PyTorch's TunableOp benchmarks the available hipBLASLt/rocBLAS kernels for each
GEMM shape and pins the winners. Tune once per machine:

```bash
./scripts/tune_rocm.sh
```

It writes `~/.cache/heartlib/tunableop_<arch>_0.csv`, which `_rocm.py` then
finds and replays automatically — no environment variables to set, and no
tuning ever happens implicitly.

| 30 s of audio | generation loop | RTF | wall clock |
|---|---|---|---|
| untuned | 9.53 it/s | 1.05 | 1 m 29 s |
| tuned | **12.96 it/s** | **0.96** | **57 s** |

Tuning must stay a deliberate, separate step. With the prefix KV cache the
attention GEMMs change shape on every frame, so a tuning run explores a far
larger space than it used to (546 entries against 4 before) and is itself
~3x slower than a normal run. The results still transfer to lengths that were
never tuned — most of the win is in the fixed-shape projections — so the
4-minute default gets 7.78 → 9.47 it/s (RTF 1.61 → 1.32) from a 20-second
tuning run.

Replay never re-tunes: shapes absent from the file fall back to the default
kernel, and a file that no longer matches the machine (PyTorch version, ROCm
version, `GCN_ARCH_NAME` are all recorded as validators) produces a
`Failed validator` warning and is ignored. Delete the cache file to re-tune.

### KV cache sizing (not AMD-specific, but it dominated everything else)

Profiling one decode frame put **84.7% of the time in the backbone** (247 ms of
292 ms) against 41 ms for the seven decoder passes — even though both use the
same layer shape and the backbone is only 9x bigger by parameter count.

The cause is in torchtune's `KVCache.update`, which returns the *entire* cache
tensor rather than a view of the filled prefix:

```python
k_out = self.k_cache        # (batch, heads, max_seq_len, head_dim)
```

`HeartMuLa.setup_caches` used to allocate that at the model's full 8192-token
context regardless of how much audio was requested, so every frame read 1.75 GB
of mostly-empty KV and softmaxed over 8192 positions that the causal mask then
threw away. Effective bandwidth was 28 GB/s against a ~256 GB/s part.

`setup_caches` now takes `max_seq_len`, and the pipeline passes
`prompt_len + max_audio_frames + 1`. Measured on gfx1151, one backbone forward:

| cache length | backbone fwd | effective bandwidth |
|---|---|---|
| 8192 (old, always) | 248 ms | 28 GB/s |
| 4096 (≈ 4 min of audio) | 150 ms | 41 GB/s |
| 1024 | 75 ms | 73 GB/s |
| 512 | 63 ms | 86 GB/s |

End-to-end for 30 s of audio: **3.42 it/s → 9.07 it/s, RTF 3.65 → 1.10**, and
wall clock 2 m 40 s → 1 m 31 s.

The positions removed were already masked out, so this is mathematically a
no-op. Verified in fp32, comparing full-context against sized-cache logits over
a prefill plus eight decode steps: max absolute difference 6.3e-5 on a logit
scale of 5.6 (relative 1.1e-5), with argmax and the full top-50 set identical.
In bf16 the same comparison drifts by ~0.2 — accumulation-order noise from
summing 8192 versus 768 terms — which top-k sampling then amplifies into a
different (equally valid) sample. Audio statistics are unchanged: RMS 0.1418
before, 0.1414 after.

Nothing about this is ROCm-specific; a CUDA run reads the same dead KV.

#### Attending only over the filled prefix

Sizing the cache to the request still pays the worst case from frame one: a
4-minute generation allocates ~3456 positions and reads all of them while
producing frame 1, when only ~400 are written.

`_PrefixKVCache` (in `modeling_heartmula.py`) subclasses torchtune's `KVCache`
and returns a view of the positions actually written; `generate_frame` narrows
the causal mask to match. Cost now tracks how far into the song we are rather
than how long the song was allowed to be.

The fill level is a Python `int` rather than `KVCache.size`, which reads
`cache_pos` off the GPU — that would be a device sync in each of the 28
backbone layers, every frame.

| | backbone fwd at position ~400 |
|---|---|
| cache allocated 768 | 68 ms → **60 ms** |
| cache allocated 3456 | 134 ms → **60 ms** |

Also equivalent rather than approximate, and checked the same way: fp32 max
absolute logit difference 8.9e-5 on a scale of 5.6, argmax and top-50 set
identical to the stock full-cache path.

#### Where the time goes now

Per frame, 30 s generation, after both fixes:

| stage | time | share |
|---|---|---|
| backbone forward | 59.8 ms | 57% |
| decoder forwards (7x) | 40.7 ms | 39% |
| sampling + embed + EOS check | 3.5 ms | 3% |

The seven decoder passes read 4.36 GB of weights per frame at ~107 GB/s, which
is at the practical ceiling for this part — they are sequential by
construction, since each codebook conditions on the previous one. Going
further would mean quantization, not scheduling.

### Performance summary

The frame rate is 12.5 Hz, so RTF 1.0 means 12.5 it/s on the generation loop.

| | 30 s of audio | 4-minute default |
|---|---|---|
| as cloned | 3.42 it/s — RTF 3.65 | 3.42 it/s — RTF 3.65 |
| KV cache sized to the request | 9.07 — RTF 1.10 | 5.69 — RTF 1.76 |
| + attend only the filled prefix | 9.53 — RTF 1.05 | 7.78 — RTF 1.61 |
| + `scripts/tune_rocm.sh` | **12.96 — RTF 0.96** | **9.47 — RTF 1.32** |

3.8x on 30-second generations and faster than real time, on an APU, from a
starting point of RTF 3.65. Two of the three steps are backend-independent
wins that a CUDA run would also see.

What is left is bandwidth. Per frame the model reads ~5.25 GB of backbone
weights plus 4.36 GB across the seven decoder passes, and those passes are
sequential by construction — each codebook conditions on the previous. The
next real step would be quantization, not scheduling.

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
