import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.emotion_detector import EmotionDetector
from preprocessing.video.face_detection import FaceDetector


EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "🙂",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}


class RealTimeEmotionApp:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or self._resolve_model_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector()
        self.emotion_model = EmotionDetector(model_path=self.model_path, device=self.device)
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.last_prediction = None
        self.last_confidence = 0.0

    def _resolve_model_path(self) -> str:
        env_path = os.getenv("EMOTION_MODEL_PATH")
        if env_path:
            return env_path

        candidates = [
            os.path.join("models", "best_scheduler_model.pth"),
            os.path.join("models", "best_emotion_model.pth"),
            os.path.join("models", "emotion_classifier.onnx"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError("No emotion model found. Set EMOTION_MODEL_PATH or place best_scheduler_model.pth in models/.")

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.last_frame_time = now

    def _draw_face_overlay(self, frame: np.ndarray, face_bbox: tuple, emotion: str, confidence: float) -> None:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        emoji = get_emotion_emoji(emotion)
        label = f"{emoji} {emotion.upper()} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._update_fps()
        detection = self.face_detector.detect(frame)

        if detection["status"] == "detected" and detection["face"] is not None:
            face_crop = detection["face"]
            prediction = self.emotion_model.predict(face_crop)
            if prediction.get("status") == "ok":
                emotion = prediction["final_emotion"]
                confidence = float(prediction["confidence"]) * 100.0

                if self.last_prediction is None or emotion != self.last_prediction or confidence >= self.last_confidence + 5.0:
                    self.last_prediction = emotion
                    self.last_confidence = confidence

                self._draw_face_overlay(frame, detection["bbox"], self.last_prediction, self.last_confidence)
                self._draw_status_panel(frame, self.last_prediction, self.last_confidence)
            else:
                self._draw_status_panel(frame, "UNKNOWN", 0.0)
        else:
            self._draw_status_panel(frame, "UNKNOWN", 0.0)

        cv2.putText(frame, f"FPS: {self.fps:0.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _draw_status_panel(self, frame: np.ndarray, emotion: str, confidence: float) -> None:
        text = build_display_text(emotion, confidence)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self, window_name: str = "Real-Time Emotion Recognition") -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam. Check camera permissions.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                processed = self.process_frame(frame)
                cv2.imshow(window_name, processed)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def get_emotion_emoji(emotion: str) -> str:
    return EMOTION_EMOJIS.get(emotion.lower(), "🙂")


def build_display_text(emotion: str, confidence: float) -> str:
    normalized_emotion = emotion.upper()
    confidence_percent = confidence if confidence >= 1 else confidence * 100.0
    return f"Current Emotion : {normalized_emotion}\nConfidence : {confidence_percent:.1f}%"


def main() -> None:
    app = RealTimeEmotionApp()
    app.run()


if __name__ == "__main__":
    main()
