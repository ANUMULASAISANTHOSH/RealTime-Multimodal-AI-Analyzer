"""
output/player.py
Collects PCM chunks from Gemini Live, wraps as WAV, base64-encodes,
and broadcasts to the browser over WebSocket.
"""
import base64, io, wave
from utils.config import OUT_SAMPLE_RATE


def _pcm_to_wav_b64(chunks: list, rate: int) -> str:
    raw = b"".join(chunks)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return base64.b64encode(buf.getvalue()).decode()


class AudioPlayer:
    def __init__(self):
        self._chunks: list = []

    def add_chunk(self, pcm_bytes: bytes) -> None:
        self._chunks.append(pcm_bytes)

    async def flush(self, transcript: str = "") -> None:
        from avatar.avatar_controller import broadcast
        if not self._chunks:
            return
        wav_b64 = _pcm_to_wav_b64(self._chunks, OUT_SAMPLE_RATE)
        await broadcast({"type": "speak", "audio": wav_b64, "transcript": transcript})
        self._chunks.clear()

    def clear(self) -> None:
        self._chunks.clear()
