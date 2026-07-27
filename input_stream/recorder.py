"""
Push-to-talk recorder.
Press Enter → speak → press Enter again → returns raw int16 PCM bytes.
No WAV header needed; the Live API takes raw PCM directly.
"""
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MIC_SAMPLE_RATE, CHANNELS


def record_until_enter() -> bytes | None:
    """
    Returns raw int16 PCM bytes of the recording, or None if the user wants to quit.
    """
    cmd = input("\nPress Enter to speak  (or type 'q' to quit): ").strip().lower()
    if cmd == "q":
        return None

    print("🎙  Recording... press Enter to stop.")
    frames = []

    def _cb(indata, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=_cb,
    ):
        input()  # callback fills `frames` while we wait

    if not frames:
        print("Nothing recorded, try again.")
        return b""

    audio = np.concatenate(frames, axis=0)
    return audio.tobytes()
