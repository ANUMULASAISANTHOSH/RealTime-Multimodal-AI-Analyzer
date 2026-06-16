from input_stream.video_stream import VideoStream
import cv2

def main():
    print("Starting video stream...")

    vs = VideoStream().start()

    while True:
        ret, frame = vs.read()

        if not ret or frame is None:
            continue

        cv2.imshow("Video Stream Test", frame)

        key = cv2.waitKey(1)
        window_closed = cv2.getWindowProperty("Video Stream Test", cv2.WND_PROP_VISIBLE) < 1

        if key in (ord('q'), 27, ord('x')) or window_closed:
            print("Exit triggered")
            break

        if key not in (-1, ord('q'), 27, ord('x')):
            print("Press 'q', ESC, or close the window to exit.")

    print("Stopping video stream...")
    vs.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()