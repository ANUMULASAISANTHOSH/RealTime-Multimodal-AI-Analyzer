"""
app/main.py - Starts WebSocket server + Gemini conversation concurrently.
Run with: python app/main.py
"""
import asyncio, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from avatar.avatar_controller import start_server
from chatbot.live_session import run_conversation


async def _main():
    print("=" * 55)
    print("  RealTime Voice Assistant  -  Browser Avatar Edition")
    print("=" * 55)
    print("Browser will open automatically. Come back here and")
    print("press Enter to speak once the avatar appears.\n")

    tasks = [
        asyncio.create_task(start_server()),
        asyncio.create_task(run_conversation()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        exc = t.exception()
        if exc:
            raise exc


def main():
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nShutting down. Bye!")


if __name__ == "__main__":
    main()
