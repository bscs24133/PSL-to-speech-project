# src/hand_crop_flow.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import atexit
import warnings
warnings.filterwarnings('ignore')

# ---------- download model file once ----------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print('Downloading MediaPipe hand landmarker model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Downloaded.')

# ---------- build detector once ----------
_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options      = mp_vision.HandLandmarkerOptions(
                    base_options=_base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                    running_mode=mp_vision.RunningMode.IMAGE
                )
_detector     = mp_vision.HandLandmarker.create_from_options(_options)


def get_hand_bbox(frame, detection_result, padding=20):
    if not detection_result.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    all_x, all_y = [], []
    for hand in detection_result.hand_landmarks:
        for lm in hand:
            all_x.append(int(lm.x * w))
            all_y.append(int(lm.y * h))
    x1 = max(min(all_x) - padding, 0)
    y1 = max(min(all_y) - padding, 0)
    x2 = min(max(all_x) + padding, w)
    y2 = min(max(all_y) + padding, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def compute_flow_with_hand_crop(frames, target_frames=16, img_size=(64, 64)):
    cropped_frames = []
    for frame in frames:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = _detector.detect(mp_image)
        bbox     = get_hand_bbox(frame, result)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
        cropped_frames.append(cv2.resize(crop, img_size))

    flows = []
    for i in range(len(cropped_frames) - 1):
        prev = cv2.cvtColor(cropped_frames[i],     cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(cropped_frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)

    while len(flows) < target_frames:
        flows.append(np.zeros((*img_size, 2), dtype=np.float32))
    flows = flows[:target_frames]

    result = np.array(flows, dtype=np.float32)
    assert result.shape == (target_frames, *img_size, 2), \
        f"Unexpected shape: {result.shape}"
    return result


def _cleanup():
    try:
        _detector.close()
    except Exception:
        pass

atexit.register(_cleanup)