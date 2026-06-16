from input_stream.video_stream import VideoStream
from preprocessing.video.face_detection import FaceDetector
import cv2


def main():
    print("Program started")
    vs = VideoStream().start()
    print("Video stream started")
    detector = FaceDetector()

    while True:
        ret, frame = vs.read()

        if not ret or frame is None:
            continue

        result = detector.detect(frame)

        if result["status"] == "detected":
            x, y, w, h = result["bbox"]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No face detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

        cv2.imshow("Face Detection Test", frame)

        key = cv2.waitKey(1)

        if key in (ord('q'), 27):
            break

    vs.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()