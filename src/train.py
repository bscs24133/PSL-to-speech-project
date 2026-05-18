# src/train.py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from config import FLOW_OUTPUT, MODEL_SAVE_DIR, BATCH_SIZE, EPOCHS, N_FRAMES, IMG_SIZE


def load_flow_data(flow_dir):
    """Load all .npy flow files from a directory of label subfolders."""
    X, y = [], []
    for label in sorted(os.listdir(flow_dir)):
        label_path = os.path.join(flow_dir, label)
        if not os.path.isdir(label_path):
            continue
        for f in os.listdir(label_path):
            if f.endswith('.npy'):
                flow = np.load(os.path.join(label_path, f))
                X.append(flow)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def build_cnn(n_classes, n_frames=N_FRAMES, img_size=IMG_SIZE):
    input_shape = (n_frames, img_size[0], img_size[1], 2)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((n_frames * img_size[0], img_size[1], 2)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),        # extra pooling layer

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # extra conv
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),   # wider dense
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),   # second dense layer
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def train_model(task='word'):
    print(f"\n{'='*50}")
    print(f"Training {task.upper()} model")
    print(f"{'='*50}")

    train_dir = os.path.join(FLOW_OUTPUT, task, 'train')
    test_dir  = os.path.join(FLOW_OUTPUT, task, 'test')

    X_train, y_train = load_flow_data(train_dir)
    X_test,  y_test  = load_flow_data(test_dir)

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Classes: {sorted(set(y_train))}")

    print("\nTrain class distribution:")
    for cls, cnt in sorted(Counter(y_train).items()):
        print(f"  {cls:30s}: {cnt}")

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    n_classes = len(le.classes_)
    print(f"\nTotal classes: {n_classes}")

    y_train_enc = tf.keras.utils.to_categorical(le.transform(y_train), n_classes)
    y_test_enc  = tf.keras.utils.to_categorical(le.transform(y_test),  n_classes)

    # Save label encoder
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_SAVE_DIR, f'{task}_label_classes.npy'), le.classes_)
    print(f"Saved {task}_label_classes.npy")

    # Class weights
    y_train_int = le.transform(y_train)
    weights = compute_class_weight('balanced',
                                   classes=np.arange(n_classes),
                                   y=y_train_int)
    class_weight_dict = dict(enumerate(weights))
    print(f"Class weights range: {weights.min():.2f} – {weights.max():.2f}")

    # Build and compile
    model = build_cnn(n_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_DIR, f'{task}_cnn.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train, y_train_enc,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_enc),
        callbacks=[checkpoint, reduce_lr, early_stop],
        class_weight=class_weight_dict
    )

    print(f"\n{task.upper()} model saved to {MODEL_SAVE_DIR}")


if __name__ == '__main__':
    train_model('word')
    train_model('alpha')