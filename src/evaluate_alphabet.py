import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from config import STATIC_TEST, MODEL_SAVE_DIR, BATCH_SIZE

datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    STATIC_TEST,
    target_size=(64,64),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(f'{MODEL_SAVE_DIR}/alphabet_cnn.h5')

y_pred = model.predict(test_gen)
y_pred = y_pred.argmax(axis=1)

y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))