# src/inference.py — STATIC (alphabet-only) inference
import cv2
import numpy as np

# ── ROI: fixed box where user places hand ─────────────────────────────────────
ROI_X1, ROI_Y1 = 320, 60
ROI_X2, ROI_Y2 = 580, 380


def get_roi_box(frame):
    return ROI_X1, ROI_Y1, ROI_X2, ROI_Y2


def preprocess_for_model(image):
    """Resize to 64x64, convert BGR→RGB, normalize."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (64, 64))
    return np.expand_dims(resized.astype("float32") / 255.0, 0)


def predict_snapshot(model, label_classes, frame):
    """
    Called on P keypress. Crops ROI, runs model, prints top 3.
    """
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    inp = preprocess_for_model(roi)
    pred = model.predict(inp, verbose=0)[0]

    top3 = np.argsort(pred)[::-1][:3]
    print("[snapshot] Top 3:")
    for i, idx in enumerate(top3):
        print(f"  Top{i+1}: {str(label_classes[idx]):15s}  conf={pred[idx]:.3f}")

    cv2.imwrite("debug_model_input_64x64.png", cv2.resize(roi, (64, 64)))
    cv2.imwrite("debug_roi.png", roi)

    best_idx = int(top3[0])
    best_conf = float(pred[best_idx])
    best_label = str(label_classes[best_idx])

    return best_label, best_conf, roi


def extract_hand_on_white(frame):
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    return roi, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2), np.ones(roi.shape[:2], np.uint8) * 255


def run_alpha_inference(model, label_classes, frame):
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    inp = preprocess_for_model(roi)
    pred = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    if conf < 0.75:
        return None
    return str(label_classes[idx])