"""
Sends the recorded audio to Gemini and gets back BOTH:
  - an exact, verbatim transcript of what was said
  - a natural conversational reply
in a single structured-output call, so the two never get mixed up or paraphrased
into each other.
"""
from google import genai
from google.genai import types
from pydantic import BaseModel

from utils.config import GEMINI_API_KEY, LIVE_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


class VoiceTurn(BaseModel):
    transcript: str
    reply: str


def get_transcript_and_reply(wav_bytes: bytes, history: list[str]) -> VoiceTurn:
    history_text = "\n".join(history) if history else "(this is the first message)"

    instructions = f"""You are a helpful, friendly voice assistant having a spoken conversation.

Conversation so far:
{history_text}

You are given a new audio clip of the user speaking. Do two things:
1. transcript: write down EXACTLY what the user said, word for word. Do not
   paraphrase, summarize, correct grammar, or add punctuation they didn't
   indicate with their pauses/tone.
2. reply: a SHORT, natural spoken reply (1-2 sentences max) responding to
   what they just said, taking the conversation history into account. Keep
   it brief - it will be read aloud and longer replies take longer to speak."""

    response = client.models.generate_content(
        model=LIVE_MODEL,
        contents=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            instructions,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VoiceTurn,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    return response.parsed
