# src/train.py
import numpy as np, os, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from config import FLOW_OUTPUT, MODEL_SAVE_DIR, BATCH_SIZE, EPOCHS

def load_flow_data(split='train'):
    X, y = [], []
    base = os.path.join(FLOW_OUTPUT, split)
    for label in os.listdir(base):
        for f in os.listdir(os.path.join(base, label)):
            if f.endswith('.npy'):
                flow = np.load(os.path.join(base, label, f))
                X.append(flow)
                y.append(label)
    return np.array(X), np.array(y)

def build_word_cnn(n_classes, input_shape=(15, 64, 64, 2)):
    model = models.Sequential([
        layers.Reshape((15*64, 64, 2), input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    X_train, y_train = load_flow_data('train')
    X_test,  y_test  = load_flow_data('test')

    print(f"Train samples: {len(X_train)}, classes: {len(set(y_train))}")
    print(f"Test samples:  {len(X_test)},  classes: {len(set(y_test))}")

    # fit on train (80 classes), test is a subset so transform works
    le = LabelEncoder()
    le.fit(y_train)
    n_classes = len(le.classes_)
    print(f"Total classes: {n_classes}")  # should be 80

    y_train_enc = tf.keras.utils.to_categorical(le.transform(y_train), num_classes=n_classes)
    y_test_enc  = tf.keras.utils.to_categorical(le.transform(y_test),  num_classes=n_classes)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_SAVE_DIR, 'label_classes.npy'), le.classes_)

    model = build_word_cnn(n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_DIR, 'word_cnn.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    model.fit(X_train, y_train_enc, batch_size=BATCH_SIZE,
            epochs=EPOCHS, validation_data=(X_test, y_test_enc),
            callbacks=[checkpoint])

