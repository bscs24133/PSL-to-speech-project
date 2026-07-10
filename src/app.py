# src/app.py — STATIC (alphabet-only) end-to-end app
import cv2
import numpy as np
import tensorflow as tf
import sys, os, time, json
from tkinter import filedialog
import tkinter as tk

sys.path.insert(0, os.path.dirname(__file__))

from translator import get_urdu_script, get_roman_urdu
from tts_engine import speak_urdu
from urdu_renderer import render_urdu_on_frame
from text_builder import TextBuilder
from inference import (predict_snapshot, get_roi_box, extract_hand_on_white,
                        ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)


def _make_frame(roi_img):
    """Wraps a ROI image into a blank frame at the correct ROI position."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h = ROI_Y2 - ROI_Y1
    w = ROI_X2 - ROI_X1
    roi_resized = cv2.resize(roi_img, (w, h))
    frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_resized
    return frame


# ── Load label order from saved JSON (guaranteed correct order) ───────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
json_path = os.path.join(MODEL_DIR, "alpha_class_indices.json")
if os.path.exists(json_path):
    with open(json_path) as f:
        idx_to_label = {v: k for k, v in json.load(f).items()}
    alpha_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    print(f"Loaded {len(alpha_labels)} alpha labels from JSON")
else:
    # Fallback hardcoded list
    alpha_labels = [
        "1-Hay", "Ain", "Alif", "Bay", "Byeh", "Chay", "Cyeh", "Daal", "Dal",
        "Dochahay", "Fay", "Gaaf", "Ghain", "Hamza", "Kaf", "Khay", "Kiaf",
        "Lam", "Meem", "Nuun", "Nuungh", "Pay", "Ray", "Say", "Seen", "Sheen",
        "Suad", "Taay", "Tay", "Tuey", "Wao", "Zaal", "Zaey", "Zay", "Zuad", "Zuey"
    ]
    print("WARNING: Using hardcoded alpha_labels — label order may be wrong!")

ALPHA_MODEL  = os.path.join(MODEL_DIR, "alphabet_cnn.h5")   # original — for file input
ALPHA_MODEL1 = os.path.join(MODEL_DIR, "alphabet_cnn1.h5")  # fine-tuned — for webcam

print("Loading models...")
alpha_model  = tf.keras.models.load_model(ALPHA_MODEL)   # file mode
alpha_model1 = tf.keras.models.load_model(ALPHA_MODEL1)  # webcam mode
print(f"Alpha classes: {len(alpha_labels)}")


# ── State ─────────────────────────────────────────────────────────────────────
text_mode           = "word"
builder             = TextBuilder(mode="word")
last_label          = ""
last_urdu           = ""
last_roman          = ""
snapshot_result     = None
snapshot_preview    = None
SNAP_DISPLAY_FRAMES = 90
snap_display_count  = 0

cap = cv2.VideoCapture(0)
time.sleep(1)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

print("\nControls:")
print("  w = Word-by-word mode       s = Sentence mode")
print("  p = Snapshot predict        f = Load image file")
print("  SPACE = commit word         ENTER = commit sentence")
print("  c = clear    q = quit\n")


def draw_ui(frame, label, urdu, roman, mode_txt, accumulated):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (170, 10), (340, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"TEXT: {mode_txt.upper()}", (175, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if label:
        cv2.putText(frame, f"Sign: {label}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Roman: {roman}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    if urdu:
        frame = render_urdu_on_frame(frame, urdu, position=(10, 130),
                                      font_size="large", color=(0, 255, 0))
    if accumulated:
        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, accumulated, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    hints = "w/s:text  p:predict  f:file  SPC/ENT:commit  c:clear  q:quit"
    cv2.putText(frame, hints, (5, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = get_roi_box(frame)

    # Yellow ROI guide box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "Fill box with ONE hand, press P",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Live ROI preview in top-right corner
    roi_preview = frame[y1:y2, x1:x2].copy()
    if roi_preview.size > 0:
        ph, pw = 150, 150
        px1 = frame.shape[1] - pw - 10
        preview = cv2.resize(roi_preview, (pw, ph))
        frame[10:10 + ph, px1:px1 + pw] = preview
        cv2.rectangle(frame, (px1, 10), (px1 + pw, 10 + ph), (255, 255, 0), 1)
        cv2.putText(frame, "Live ROI", (px1, 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Show snapshot result
    if snap_display_count > 0:
        snap_display_count -= 1
        if snapshot_preview is not None and snapshot_preview.size > 0:
            px1 = frame.shape[1] - 160
            sp = cv2.resize(snapshot_preview, (150, 150))
            if frame.shape[0] > 315:
                frame[165:315, px1:px1 + 150] = sp
                cv2.putText(frame, "Last snapshot", (px1, 163),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        if snapshot_result:
            lbl, conf = snapshot_result
            color = (0, 255, 0) if conf >= 0.60 else (0, 165, 255)
            cv2.putText(frame, f">> {lbl} ({conf:.2f})",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    frame = draw_ui(frame, last_label, last_urdu, last_roman,
                     text_mode, builder.get_display())
    cv2.imshow("PSL to Speech", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("f"):
        # Open file dialog to pick an image
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select sign image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()

        if file_path:
            print(f"[file] Loading: {file_path}")
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                small = cv2.resize(img, (64, 64))
                inp = small.astype("float32") / 255.0
                inp = np.expand_dims(inp, 0)
                pred = alpha_model.predict(inp, verbose=0)[0]
                top3 = np.argsort(pred)[::-1][:3]
                print("[file] Top 3 by INDEX:")
                for i, idx in enumerate(top3):
                    print(f"  Top{i+1}: idx={idx}  label={alpha_labels[idx]}  conf={pred[idx]:.3f}")
                label = alpha_labels[int(top3[0])]
                conf = float(pred[top3[0]])
                snapshot_preview = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (150, 150))
                snap_display_count = SNAP_DISPLAY_FRAMES
                snapshot_result = (label, conf)
                last_label = label
                last_urdu = get_urdu_script(label, "urdu_alpha")
                last_roman = get_roman_urdu(label, "urdu_alpha")
                builder.add(label)
                speak_urdu(last_urdu)
                print(f"[file] → {label}  conf={conf:.2f}")
            else:
                print("[file] Failed to load image")

    elif key == ord("w"):
        text_mode = "word"
        builder.set_mode("word")
        print("Word mode")

    elif key == ord("s"):
        text_mode = "sentence"
        builder.set_mode("sentence")
        print("Sentence mode")

    elif key == ord("c"):
        builder.clear()
        last_label = last_urdu = last_roman = ""
        snapshot_result = None
        snap_display_count = 0
        print("Cleared")

    elif key == ord("p"):
        print("[snapshot] Capturing...")
        label, conf, snap_img = predict_snapshot(alpha_model1, alpha_labels, frame)  # webcam model
        snapshot_preview = snap_img
        snap_display_count = SNAP_DISPLAY_FRAMES
        snapshot_result = (label, conf)

        # Accept all predictions — let user decide via SPACE/ENTER
        last_label = label
        last_urdu = get_urdu_script(label, "urdu_alpha")
        last_roman = get_roman_urdu(label, "urdu_alpha")
        builder.add(label)
        if text_mode == "word":
            speak_urdu(last_urdu)
        print(f"[snapshot] Accepted: {label}  conf={conf:.2f}")

    elif key == 32:  # SPACE
        result = builder.commit()
        if result:
            urdu = get_urdu_script(result, "urdu_alpha")
            speak_urdu(urdu)
            print(f"Word committed: {result} → {urdu}")

    elif key == 13:  # ENTER
        result = builder.commit()
        if result:
            words = result.split()
            urdu_sent = " ".join(get_urdu_script(w, "urdu_alpha") for w in words)
            last_urdu = urdu_sent
            speak_urdu(urdu_sent)
            print(f"Sentence: {result} → {urdu_sent}")

cap.release()
cv2.destroyAllWindows()
print("App closed.")