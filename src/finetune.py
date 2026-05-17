# src/finetune.py
import tensorflow as tf
import numpy as np
import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
WEBCAM_DIR = "webcam_train"

# ── Load existing model ───────────────────────────────────────────────────────
print("Loading existing model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "alphabet_cnn.h5"))

# ── Freeze first 10 layers, retrain only the last layers ─────────────────────
for layer in model.layers:
    layer.trainable = True

print(f"All {len(model.layers)} layers trainable")

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable} / {len(model.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # ← lower from 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Load class order from saved JSON ─────────────────────────────────────────
json_path = os.path.join(MODEL_DIR, "alpha_class_indices.json")
with open(json_path) as f:
    class_indices = json.load(f)
ordered_classes = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
print(f"Classes: {len(ordered_classes)}")

# Check what webcam data exists
webcam_classes = sorted(os.listdir(WEBCAM_DIR))
print(f"Webcam data collected for: {webcam_classes}")
total_samples = sum(
    len(os.listdir(os.path.join(WEBCAM_DIR, c)))
    for c in webcam_classes
)
print(f"Total webcam samples: {total_samples}")

# ── Data generators ───────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,   # ← False! PSL signs are not symmetric
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    WEBCAM_DIR,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='training',
    classes=ordered_classes   # must match original training order
)
val_gen = datagen.flow_from_directory(
    WEBCAM_DIR,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    classes=ordered_classes
)

print(f"\nTrain batches: {len(train_gen)} | Val batches: {len(val_gen)}")
print("Starting fine-tuning...\n")

# ── Train ─────────────────────────────────────────────────────────────────────
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "alphabet_cnn1.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    train_gen,
    epochs=40,              # ← increase from 20
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

print("\nFine-tuning complete! Model saved.")
print("Run: python src/app.py")