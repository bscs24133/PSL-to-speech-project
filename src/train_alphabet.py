# src/train_alphabet.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import STATIC_TRAIN, STATIC_TEST, MODEL_SAVE_DIR, BATCH_SIZE, EPOCHS
import os

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
datagen_test = ImageDataGenerator(rescale=1./255)

train_gen = datagen_train.flow_from_directory(
    STATIC_TRAIN, target_size=(64, 64), batch_size=BATCH_SIZE, class_mode='categorical')
test_gen = datagen_test.flow_from_directory(
    STATIC_TEST, target_size=(64, 64), batch_size=BATCH_SIZE, class_mode='categorical')

def build_alphabet_cnn(n_classes=36):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

n_classes = train_gen.num_classes
print(f"Number of classes: {n_classes}")
model = build_alphabet_cnn(n_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'{MODEL_SAVE_DIR}/alphabet_cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=[checkpoint, reduce_lr]
)
print('Alphabet CNN saved.')
