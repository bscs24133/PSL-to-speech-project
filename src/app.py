# src/app.py
# Final Phase 3 app — live PSL recognition with Urdu translation and TTS

import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import time
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from translator import get_urdu_script, get_roman_urdu
from tts_engine import speak_urdu
from urdu_renderer import render_urdu_on_frame
from text_builder import TextBuilder
from inference import run_word_inference, run_alpha_inference, WINDOW_SIZE

# ── Hardcoded alpha labels (no USB needed) ───────────────────────────────────
alpha_labels = [
    "1-Hay","Ain","Alif","Bay","Byeh","Chay","Cyeh","Daal","Dal",
    "Dochahay","Fay","Gaaf","Ghain","Hamza","Kaf","Khay","Kiaf",
    "Lam","Meem","Nuun","Nuungh","Pay","Ray","Say","Seen","Sheen",
    "Suad","Taay","Tay","Tuey","Wao","Zaal","Zaey","Zay","Zuad","Zuey"
]

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
WORD_MODEL  = os.path.join(MODEL_DIR, "word_cnn.h5")
ALPHA_MODEL = os.path.join(MODEL_DIR, "alphabet_cnn.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.npy")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
word_model  = tf.keras.models.load_model(WORD_MODEL)
alpha_model = tf.keras.models.load_model(ALPHA_MODEL)
word_labels = np.load(LABELS_PATH, allow_pickle=True)
print(f"Word classes: {len(word_labels)}  |  Alpha classes: {len(alpha_labels)}")

# ── State ─────────────────────────────────────────────────────────────────────
input_mode       = "video"
text_mode        = "word"
builder          = TextBuilder(mode="word")
frame_buffer     = deque(maxlen=WINDOW_SIZE)
last_label       = ""
last_urdu        = ""
last_roman       = ""
stable_label     = ""
stable_count     = 0
STABLE_THRESHOLD = 5


# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
time.sleep(2)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

print("\nControls:")
print("  i = Image mode (alphabet)   v = Video mode (words)")
print("  w = Word-by-word mode       s = Sentence mode")
print("  SPACE = commit word         ENTER = commit sentence")
print("  c = clear buffer            q = quit\n")


def draw_ui(frame, label, urdu, roman, mode_in, mode_txt, accumulated):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (10, 10), (160, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"MODE: {mode_in.upper()}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.rectangle(frame, (170, 10), (340, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"TEXT: {mode_txt.upper()}", (175, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if label:
        cv2.putText(frame, f"Sign: {label}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Roman: {roman}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    if urdu:
        frame = render_urdu_on_frame(frame, urdu,
                                     position=(10, 130),
                                     font_size="large",
                                     color=(0, 255, 0))

    if accumulated:
        cv2.rectangle(frame, (0, h-50), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, accumulated, (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    hints = "i/v:mode  w/s:text  SPC/ENT:commit  c:clear  q:quit"
    cv2.putText(frame, hints, (5, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    predicted_label = None

    # ── Inference ─────────────────────────────────────────────────────────
    if input_mode == "image":
        predicted_label = run_alpha_inference(alpha_model, alpha_labels, frame)
    else:
        predicted_label = run_word_inference(word_model, word_labels,
                                             frame_buffer, frame)

    # ── Stabilizer ────────────────────────────────────────────────────────
    if predicted_label:
        if predicted_label == stable_label:
            stable_count += 1
        else:
            stable_label = predicted_label
            stable_count = 1

        if stable_count == STABLE_THRESHOLD and stable_label != last_label:
            last_label = stable_label
            mode = "urdu_alpha" if input_mode == "image" else "word"
            last_urdu  = get_urdu_script(last_label, mode)
            last_roman = get_roman_urdu(last_label, mode)
            builder.add(last_label)
            if text_mode == "word":
                speak_urdu(last_urdu)
    else:
        stable_count = 0

    # ── Draw UI ───────────────────────────────────────────────────────────
    frame = draw_ui(frame, last_label, last_urdu, last_roman,
                    input_mode, text_mode, builder.get_display())

    cv2.imshow("PSL to Speech", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("i"):
        input_mode = "image"
        frame_buffer.clear()
        builder.clear()
        last_label = last_urdu = last_roman = ""
        stable_label = ""; stable_count = 0
        print("Switched to IMAGE mode")
    elif key == ord("v"):
        input_mode = "video"
        frame_buffer.clear()
        builder.clear()
        last_label = last_urdu = last_roman = ""
        stable_label = ""; stable_count = 0
        print("Switched to VIDEO mode")
    elif key == ord("w"):
        text_mode = "word"
        builder.set_mode("word")
        print("Switched to WORD mode")
    elif key == ord("s"):
        text_mode = "sentence"
        builder.set_mode("sentence")
        print("Switched to SENTENCE mode")
    elif key == ord("c"):
        builder.clear()
        frame_buffer.clear()
        last_label = last_urdu = last_roman = ""
        stable_label = ""; stable_count = 0
        print("Buffer cleared")
    elif key == 32:
        result = builder.commit()
        if result:
            mode = "urdu_alpha" if input_mode == "image" else "word"
            urdu = get_urdu_script(result, mode)
            speak_urdu(urdu)
            print(f"Word committed: {result} → {urdu}")
    elif key == 13:
        result = builder.commit()
        if result:
            mode = "urdu_alpha" if input_mode == "image" else "word"
            words = result.split()
            urdu_words = [get_urdu_script(w, mode) for w in words]
            urdu_sentence = " ".join(urdu_words)
            last_urdu = urdu_sentence
            speak_urdu(urdu_sentence)
            print(f"Sentence committed: {result} → {urdu_sentence}")

cap.release()
cv2.destroyAllWindows()
print("App closed.")
