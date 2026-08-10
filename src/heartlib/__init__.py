from ._rocm import configure_rocm_defaults

# Must run before the first convolution reaches MIOpen; a no-op off ROCm.
configure_rocm_defaults()

from .pipelines.music_generation import HeartMuLaGenPipeline
from .pipelines.lyrics_transcription import HeartTranscriptorPipeline

__all__ = [
    "HeartMuLaGenPipeline",
    "HeartTranscriptorPipeline"
]
