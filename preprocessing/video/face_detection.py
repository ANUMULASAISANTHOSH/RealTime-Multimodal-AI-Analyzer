import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        h, w, _ = frame.shape

        # BGR → RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if not results.detections:
            return {
                "face": None,
                "bbox": None,
                "status": "no_face"
            }

        faces = []

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Clamp inside frame
            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)

            # Face center
            face_center_x = x + bw // 2
            face_center_y = y + bh // 2

            # Frame center
            frame_center_x = w // 2
            frame_center_y = h // 2

            distance = np.sqrt(
                (face_center_x - frame_center_x) ** 2 +
                (face_center_y - frame_center_y) ** 2
            )

            area = bw * bh

            faces.append({
                "bbox": (x, y, bw, bh),
                "area": area,
                "distance": distance
            })

        # 🔥 Pick best face
        faces.sort(key=lambda f: (-f["area"], f["distance"]))
        best_face = faces[0]

        x, y, bw, bh = best_face["bbox"]

        # 🔥 IMPORTANT FIX: MORE PADDING FOR 224 MODEL
        padding_x = int(0.35 * bw)
        padding_y = int(0.45 * bh)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(w, x + bw + padding_x)
        y2 = min(h, y + bh + padding_y)

        face_crop = frame[y1:y2, x1:x2]

        # 🔥 SAFETY CHECK (avoid empty crops)
        if face_crop.size == 0:
            return {
                "face": None,
                "bbox": None,
                "status": "no_face"
            }

        return {
            "face": face_crop,
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "status": "detected"
        }