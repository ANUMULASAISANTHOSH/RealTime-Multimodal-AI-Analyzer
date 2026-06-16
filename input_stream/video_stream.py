import cv2
import threading
import time


class VideoStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.ret = False
        self.frame = None
        self.stopped = False

        self.lock = threading.Lock()

    def start(self):
        """Start the video capture thread"""
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Continuously capture frames"""
        while not self.stopped:
            ret, frame = self.cap.read()

            if not ret:
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame

            time.sleep(1 / self.fps)

    def read(self):
        """Return the latest frame"""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop the stream"""
        self.stopped = True
        self.cap.release()