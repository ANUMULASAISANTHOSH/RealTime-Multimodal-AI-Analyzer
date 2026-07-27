from app.realtime_emotion_app import build_display_text, get_emotion_emoji


def test_build_display_text_formats_emotion_and_confidence():
    text = build_display_text("happy", 0.986)
    assert "HAPPY" in text
    assert "98.6%" in text


def test_get_emotion_emoji_returns_expected_symbol():
    assert get_emotion_emoji("happy") == "🙂"
    assert get_emotion_emoji("sad") == "😢"
