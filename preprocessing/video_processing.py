# Video Processing Module
# Converts video → frames for further processing

import cv2
from preprocessing.image_processing import preprocess_image


# 1️⃣ Extract Frames from Video
def extract_frames(video_path):
    """
    Reads a video and extracts frames one by one
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


# 2️⃣ Extract Frames with Interval (Optimized)
def extract_frames_interval(video_path, frame_skip=30):
    """
    Extract frames at intervals (e.g., 1 frame per second)
    frame_skip=30 → if video is 30fps
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_skip == 0:
            frames.append(frame)

        count += 1

    cap.release()

    return frames


# 3️⃣ Preprocess Video Frames (Using Image Pipeline)
def preprocess_video(video_path):
    """
    Full pipeline:
    video → frames → preprocess each frame → tensors
    """

    cap = cv2.VideoCapture(video_path)

    processed_frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, (224, 224))

        # Normalize
        frame = frame / 255.0

        processed_frames.append(frame)

    cap.release()

    return processed_frames


# 🧪 Testing the Video Pipeline
if __name__ == "__main__":

    video_path = "data/videos/test.mp4"

    frames = extract_frames_interval(video_path)

    print("Total Frames Extracted:", len(frames))