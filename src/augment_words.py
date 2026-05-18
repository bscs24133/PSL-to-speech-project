# src/augment_words.py
import cv2
import numpy as np
import os
import random
from config import WORD_TRAIN, WORD_TARGET_PER_CLASS


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_frames_as_video(frames, path, fps=15):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def augment_frames(frames):
    """
    Pick one random augmentation and apply it to all frames.
    Returns augmented frame list.
    """
    if not frames:
        return frames

    choice = random.randint(0, 4)
    augmented = []

    for frame in frames:
        f = frame.copy()

        if choice == 0:
            # Horizontal flip
            f = cv2.flip(f, 1)

        elif choice == 1:
            # Brightness jitter
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= random.uniform(0.7, 1.3)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif choice == 2:
            # Slight zoom in (crop center and resize back)
            h, w = f.shape[:2]
            scale = random.uniform(0.80, 0.92)
            cx, cy = w // 2, h // 2
            half_w = int(cx * scale)
            half_h = int(cy * scale)
            x1 = max(cx - half_w, 0)
            y1 = max(cy - half_h, 0)
            x2 = min(cx + half_w, w)
            y2 = min(cy + half_h, h)
            cropped = f[y1:y2, x1:x2]
            f = cv2.resize(cropped, (w, h))

        elif choice == 3:
            # Gaussian noise
            noise = np.random.normal(0, 8, f.shape).astype(np.int16)
            f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif choice == 4:
            # Slight rotation (-10 to +10 degrees)
            h, w = f.shape[:2]
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            f = cv2.warpAffine(f, M, (w, h))

        augmented.append(f)

    # Temporal augmentation: randomly reverse half the time on top of spatial aug
    if random.random() < 0.3:
        augmented = augmented[::-1]

    return augmented


def augment_word_dataset():
    print("=" * 50)
    print("Augmenting WORD_TRAIN dataset")
    print(f"Target per class: {WORD_TARGET_PER_CLASS}")
    print("=" * 50)

    for label in sorted(os.listdir(WORD_TRAIN)):
        label_path = os.path.join(WORD_TRAIN, label)
        if not os.path.isdir(label_path):
            continue

        videos = [v for v in os.listdir(label_path) if v.endswith('.mp4')]
        original_count = len(videos)

        if original_count == 0:
            print(f"  {label}: no videos found, skipping")
            continue

        if original_count >= WORD_TARGET_PER_CLASS:
            print(f"  {label}: already has {original_count} videos, skipping")
            continue

        needed = WORD_TARGET_PER_CLASS - original_count
        aug_count = 0

        while aug_count < needed:
            src_vid = random.choice(videos)
            src_path = os.path.join(label_path, src_vid)
            frames = read_video_frames(src_path)

            if len(frames) < 2:
                continue

            aug_frames = augment_frames(frames)
            out_name = f"aug_{aug_count:03d}_{src_vid}"
            out_path = os.path.join(label_path, out_name)
            save_frames_as_video(aug_frames, out_path)
            aug_count += 1

        print(f"  {label}: {original_count} original + {aug_count} augmented = {original_count + aug_count} total")

    print("\nAugmentation complete.")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    augment_word_dataset()