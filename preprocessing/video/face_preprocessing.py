import cv2
import numpy as np


class FacePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess(self, face):
        if face is None:
            return {
                "processed_face": None,
                "status": "no_face"
            }

        try:
            # 🔹 Step 1: Resize
            face_resized = cv2.resize(face, self.target_size)

            # 🔹 Step 2: Convert BGR → RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # 🔹 Step 3: Normalize (0–255 → 0–1)
            face_normalized = face_rgb / 255.0

            return {
                "processed_face": face_normalized,
                "status": "ok"
            }

        except Exception as e:
            return {
                "processed_face": None,
                "status": "error"
            }