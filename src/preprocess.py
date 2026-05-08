# src/preprocess.py
import cv2
import os
import numpy as np
from config import WORD_TRAIN, WORD_TEST, N_FRAMES, IMG_SIZE

def extract_frames(video_path, n=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame)
    cap.release()
    return frames

def augment_frames(frames):
    augmented = [frames]
    augmented.append([cv2.flip(f, 1) for f in frames])
    bright = [cv2.convertScaleAbs(f, alpha=1.2, beta=20) for f in frames]
    augmented.append(bright)
    noisy = [f + np.random.normal(0, 10, f.shape).astype(np.uint8) for f in frames]
    augmented.append(noisy)
    return augmented

if __name__ == '__main__':
    import glob
    videos = glob.glob(WORD_TRAIN + '/**/*.mp4', recursive=True)
    print(f'Found {len(videos)} training videos')
    frames = extract_frames(videos[0])
    print(f'Extracted {len(frames)} frames, shape: {frames[0].shape}')
