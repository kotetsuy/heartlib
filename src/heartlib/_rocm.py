"""ROCm runtime defaults.

Applied at import time, before any convolution reaches MIOpen. Everything here
is a *default*: an environment variable the user already set always wins, and
none of it runs on a non-ROCm build.
"""

import os


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
