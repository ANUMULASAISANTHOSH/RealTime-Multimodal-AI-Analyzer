"""
Converts text into speech using Gemini's native TTS model and plays it back
using sounddevice (the same audio library already used for recording).
"""
import numpy as np
import sounddevice as sd

from google import genai
from google.genai import types

from utils.config import GEMINI_API_KEY, LIVE_MODEL, VOICE_NAME

client = genai.Client(api_key=GEMINI_API_KEY)

TTS_SAMPLE_RATE = 24000  # Gemini's TTS output is fixed at 24kHz, mono, 16-bit PCM


def speak(text: str) -> None:
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            ),
        ),
    )

    # Look through all parts for the audio one, instead of assuming it's parts[0]
    pcm_data = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            pcm_data = part.inline_data.data
            break

    if not pcm_data:
        print("No audio came back from Gemini for this reply - skipping playback.")
        return

    audio = np.frombuffer(pcm_data, dtype=np.int16)
    sd.play(audio, samplerate=TTS_SAMPLE_RATE)
    sd.wait()  # blocks until playback finishes, same behavior as before