"""
utils/config.py - Centralized configuration.
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit("GEMINI_API_KEY missing. Add it to your .env file.")

LIVE_MODEL      = "gemini-3.1-flash-live-preview"
VOICE_NAME      = "Kore"
MIC_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 24000
CHANNELS        = 1
WS_PORT         = 7861
