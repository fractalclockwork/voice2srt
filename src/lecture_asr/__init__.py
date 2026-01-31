# src/lecture_asr/__init__.py

"""
lecture_asr
-----------

Local ASR -> sentence‑aligned subtitle generation using Whisper Medium.
"""

from .pipeline import transcribe
from .segmentation import sentence_segment, split_for_readability
from .srt import write_srt

__all__ = [
    "transcribe",
    "sentence_segment",
    "split_for_readability",
    "write_srt",
]
