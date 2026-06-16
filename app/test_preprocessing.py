from input_stream.video_stream import VideoStream
from preprocessing.video.face_detection import FaceDetector
from preprocessing.video.face_preprocessing import FacePreprocessor
import cv2


def main():
    vs = VideoStream().start()
    detector = FaceDetector()
    preprocessor = FacePreprocessor()

    while True:
        ret, frame = vs.read()

        if not ret or frame is None:
            continue

        result = detector.detect(frame)

        if result["status"] == "detected":
            x, y, w, h = result["bbox"]
            face = result["face"]

            processed = preprocessor.preprocess(face)

            if processed["status"] == "ok":
                # Show original bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Show processed face
                face_display = (processed["processed_face"] * 255).astype("uint8")
                cv2.imshow("Processed Face", face_display)

        else:
            cv2.putText(frame, "No face detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

        cv2.imshow("Main Frame", frame)

        if cv2.waitKey(1) in (ord('q'), 27):
            break

    vs.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()