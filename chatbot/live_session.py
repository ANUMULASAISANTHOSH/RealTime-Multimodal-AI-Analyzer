"""
chatbot/live_session.py
Gemini Live session - pushes status, transcripts and audio to browser.
"""
import asyncio
from google import genai
from google.genai import types
from utils.config import GEMINI_API_KEY, LIVE_MODEL, VOICE_NAME, MIC_SAMPLE_RATE
from output.player import AudioPlayer
from avatar.avatar_controller import broadcast

SYSTEM_PROMPT = """You are a helpful, friendly voice assistant having a natural spoken conversation.
Keep replies concise and conversational - 1-3 sentences max - they will be spoken aloud."""


async def run_conversation() -> None:
    client = genai.Client(api_key=GEMINI_API_KEY)
    player = AudioPlayer()

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
    )

    print(f"Connected to Gemini Live ({LIVE_MODEL})")

    async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
        from input_stream.recorder import record_until_enter

        while True:
            await broadcast({"type": "status", "value": "listening"})

            pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                None, record_until_enter
            )

            if pcm_bytes is None:
                print("\nGoodbye!")
                break
            if not pcm_bytes:
                continue

            await broadcast({"type": "status", "value": "thinking"})
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={MIC_SAMPLE_RATE}")
            )
            await session.send_realtime_input(activity_end=types.ActivityEnd())

            player.clear()
            output_transcript = ""

            async for msg in session.receive():
                sc = msg.server_content
                if sc is None:
                    continue

                if sc.input_transcription and sc.input_transcription.text:
                    text = sc.input_transcription.text
                    print(f"\nYou said : {text}", flush=True)
                    await broadcast({"type": "transcript", "speaker": "user", "text": text})

                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            player.add_chunk(part.inline_data.data)

                if sc.output_transcription and sc.output_transcription.text:
                    output_transcript += sc.output_transcription.text

                if sc.turn_complete:
                    print(f"Assistant: {output_transcript}", flush=True)
                    await player.flush(transcript=output_transcript)
                    break
