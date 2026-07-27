import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import timm
import torch
from fastapi import HTTPException
from torchvision import transforms as T

from preprocessing.video.face_detection import FaceDetector


class EmotionInferenceService:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_model_path(model_path)
        self.face_detector = FaceDetector()
        self.model = None
        self._load_model()

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        env_path = os.getenv("EMOTION_MODEL_PATH")
        if env_path:
            candidates.append(Path(env_path))

        project_root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                project_root / "models" / "best_scheduler_model.pth",
                project_root / "best_scheduler_model.pth",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Emotion checkpoint not found. Place best_scheduler_model.pth in models/ or set EMOTION_MODEL_PATH."
        )

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(self.emotions))
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if isinstance(state_dict, torch.nn.Module):
            self.model = state_dict.to(self.device).eval()
            return

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace("module.", "", 1).replace("model.", "", 1)
            cleaned_state_dict[cleaned_key] = value

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.to(self.device).eval()

    def preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Face crop is empty")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))

        image_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(image_tensor).unsqueeze(0).to(self.device)

    def predict_from_image(self, image_bgr: np.ndarray) -> Dict[str, object]:
        detection = self.face_detector.detect(image_bgr)
        if detection["status"] != "detected" or detection["face"] is None:
            raise HTTPException(status_code=404, detail="No face detected in the uploaded image.")

        face_crop = detection["face"]
        tensor = self.preprocess_face(face_crop)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probabilities = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
        top_idx = int(np.argmax(probs))

        return {
            "emotion": self.emotions[top_idx],
            "confidence": round(float(probs[top_idx]), 6),
            "probabilities": probabilities,
        }
