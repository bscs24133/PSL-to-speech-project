# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.metrics import classification_report, confusion_matrix
from train import load_flow_data
from config import MODEL_SAVE_DIR

TASK = "alpha"   # change to "word" if needed
from config import FLOW_OUTPUT

# Load model + correct label classes
model_path = os.path.join(MODEL_SAVE_DIR, f"{TASK}_cnn.h5")
classes_path = os.path.join(MODEL_SAVE_DIR, f"{TASK}_label_classes.npy")

model = tf.keras.models.load_model(model_path)
classes = np.load(classes_path, allow_pickle=True)

# Load test data
X_test, y_test = load_flow_data(os.path.join(FLOW_OUTPUT, TASK, "test"))


# Encode true labels using same class order as training
class_to_idx = {c: i for i, c in enumerate(classes)}
y_true = np.array([class_to_idx[l] for l in y_test])

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

labels = list(range(len(classes)))

# Report
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=classes,
    zero_division=0
))

# Confusion matrix
os.makedirs("outputs", exist_ok=True)

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title(f"{TASK.upper()} CNN Confusion Matrix")
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

print("Confusion matrix saved to outputs/confusion_matrix.png")