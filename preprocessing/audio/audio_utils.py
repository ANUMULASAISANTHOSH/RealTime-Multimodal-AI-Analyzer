"""
Audio format helpers: wrap recorded samples as WAV bytes Gemini can accept,
and wrap Gemini's raw PCM TTS output as a playable WAV.
"""
import io
import wave

import numpy as np

from utils.config import MIC_SAMPLE_RATE, CHANNELS


def numpy_to_wav_bytes(audio: np.ndarray, samplerate: int =  MIC_SAMPLE_RATE) -> bytes:
    """Wrap raw int16 mic samples in a WAV header, entirely in memory (no temp files)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 -> 2 bytes per sample
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()


def pcm_to_wav_bytes(pcm_data: bytes, samplerate: int = 24000) -> bytes:
    """Gemini's TTS model returns raw 16-bit PCM at 24kHz; wrap it as WAV for playback."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()
