"""Audio file output.

``torchaudio.save`` is the obvious way to write the generated waveform, but
from torchaudio 2.9 it delegates to TorchCodec, which is not part of the ROCm
wheel set — the call raises ``ImportError: TorchCodec is required``. soundfile
is already a dependency and libsndfile can encode mp3/flac/wav, so it makes a
solid fallback that keeps the same output paths working on every backend.
"""

import torch


def save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Write ``wav`` to ``path``.

    Args:
        path: Output file. The extension selects the container/codec.
        wav: Waveform shaped ``(channels, samples)``, any dtype/device.
        sample_rate: Sample rate in Hz.
    """
    wav = wav.detach().to(torch.float32).cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    try:
        import torchaudio

        torchaudio.save(path, wav, sample_rate)
        return
    except Exception as torchaudio_error:
        try:
            import soundfile as sf
        except ImportError:
            raise torchaudio_error

        # soundfile expects (samples, channels).
        sf.write(path, wav.transpose(0, 1).numpy(), sample_rate)
