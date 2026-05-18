# src/preprocess.py
import cv2
from config import IMG_SIZE


def extract_frames(video_path, img_size=IMG_SIZE):
    """Read all frames from a video, resize, return as list."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)
    cap.release()
    return frames