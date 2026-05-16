# src/inference.py
# Sliding window inference from webcam for word and alphabet recognition

import cv2
import numpy as np
from collections import deque
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from optical_flow import compute_flow

WINDOW_SIZE = 16  # must match N_FRAMES in config.py


def preprocess_frame(frame, img_size=(64, 64)) -> np.ndarray:
    """Resize and normalize a single frame."""
    return cv2.resize(frame, img_size)


def run_word_inference(model, label_classes, frame_buffer: deque,
                       new_frame: np.ndarray) -> str | None:
    """
    Video mode — sliding window over optical flow.
    Returns predicted label or None if buffer not full yet.
    """
    small_frame = preprocess_frame(new_frame)
    frame_buffer.append(small_frame)

    if len(frame_buffer) < WINDOW_SIZE:
        return None

    frames = list(frame_buffer)
    flows = []
    for i in range(len(frames) - 1):
        flow = compute_flow([frames[i], frames[i+1]])
        flows.append(flow[0])  # compute_flow returns array of flows

    flow_stack = np.stack(flows, axis=0)          # (15, 64, 64, 2)
    flow_input = np.expand_dims(flow_stack, 0)     # (1, 15, 64, 64, 2)

    pred = model.predict(flow_input, verbose=0)
    idx = np.argmax(pred)
    confidence = pred[0][idx]

    if confidence < 0.75:
        return None

    return str(label_classes[idx])


def run_alpha_inference(model, label_classes, frame: np.ndarray) -> str | None:
    """
    Image mode — single frame to alphabet CNN.
    Returns predicted label or None if confidence too low.
    """
    small = cv2.resize(frame, (64, 64))
    inp = small.astype("float32") / 255.0
    inp = np.expand_dims(inp, 0)                   # (1, 224, 224, 3)

    pred = model.predict(inp, verbose=0)
    idx = np.argmax(pred)
    confidence = pred[0][idx]

    if confidence < 0.75:
        return None

    return str(label_classes[idx])
