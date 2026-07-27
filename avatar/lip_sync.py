"""
avatar/lip_sync.py

Converts raw PCM audio bytes into a per-frame amplitude envelope (0.0 – 1.0).
The avatar controller uses this list to know how wide to open the mouth
on each display frame while the audio is playing.
"""
import numpy as np


def compute_envelope(pcm_bytes: bytes, sample_rate: int = 24000, fps: int = 30) -> list[float]:
    """
    Split PCM audio into fps-sized windows and compute the RMS amplitude
    of each window, normalized and smoothed into a 0.0–1.0 envelope.

    Args:
        pcm_bytes:   Raw int16 PCM from Gemini (24kHz, mono).
        sample_rate: Gemini TTS output sample rate (fixed at 24000).
        fps:         Display frames per second — one envelope value per frame.

    Returns:
        List of float values, one per frame. 0.0 = silence, 1.0 = loudest.
    """
    if not pcm_bytes:
        return [0.0]

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples_per_frame = max(1, sample_rate // fps)

    envelope = []
    for i in range(0, len(samples), samples_per_frame):
        chunk = samples[i : i + samples_per_frame]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
        envelope.append(float(rms))

    if not envelope:
        return [0.0]

    # Normalize to 0–1
    peak = max(envelope) or 1.0
    envelope = [v / peak for v in envelope]

    # Simple smoothing so the mouth doesn't flicker
    smoothed = envelope[:]
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = 0.25 * envelope[i - 1] + 0.5 * envelope[i] + 0.25 * envelope[i + 1]

    return smoothed
