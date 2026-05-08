# src/train_alphabet.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import STATIC_TRAIN, STATIC_TEST, MODEL_SAVE_DIR, BATCH_SIZE, EPOCHS
import os

datagen_train = ImageDataGenerator(
    rescale=1./255, rotation_range=15,
    width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, brightness_range=[0.8, 1.2]
)
datagen_test = ImageDataGenerator(rescale=1./255)

train_gen = datagen_train.flow_from_directory(
    STATIC_TRAIN, target_size=(224,224), batch_size=BATCH_SIZE, class_mode='categorical')
test_gen = datagen_test.flow_from_directory(
    STATIC_TEST, target_size=(224,224), batch_size=BATCH_SIZE, class_mode='categorical')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=20, validation_data=test_gen)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model.save(f'{MODEL_SAVE_DIR}/alphabet_cnn.h5')
print('Alphabet CNN saved.')