import os

import cv2

from input_stream.video_stream import VideoStream
from preprocessing.video.face_detection import FaceDetector
from models.emotion_detector import EmotionDetector


def main():
    vs = VideoStream().start()
    detector = FaceDetector()

    model_path = "models/best_emotion_model.pth"
    if not os.path.exists(model_path):
        model_path = "models/emotion_classifier.onnx"

    emotion_model = EmotionDetector(model_path=model_path)

    frame_count = 0

    while True:
        ret, frame = vs.read()

        if not ret or frame is None:
            continue

        result = detector.detect(frame)

        if result["status"] == "detected":
            x, y, w, h = result["bbox"]
            face = result["face"]

            # 🔥 DEBUG FACE (VERY IMPORTANT)
            cv2.imshow("face_debug", face)

            # 🔥 OPTIONAL: skip very small faces
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            frame_count += 1

            # 🔥 reduce flicker (process every 3rd frame)
            if frame_count % 3 == 0:
                emotion = emotion_model.predict(face)

                if emotion["status"] == "ok":
                    text = f'{emotion["final_emotion"]} ({emotion["confidence"]:.2f})'

                    # 🔥 TEXT DISPLAY (improved visibility)
                    cv2.putText(
                        frame,
                        text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

            # 🔥 DRAW FACE BOX
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # 🔥 MAIN WINDOW
        cv2.imshow("Emotion Detection (ConvNeXt / GPU)", frame)

        # 🔥 EXIT
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()