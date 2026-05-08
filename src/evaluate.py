# src/evaluate.py
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf, os
from config import MODEL_SAVE_DIR
from train import load_flow_data

model = tf.keras.models.load_model(f'{MODEL_SAVE_DIR}/word_cnn.h5')
classes = np.load(f'{MODEL_SAVE_DIR}/label_classes.npy', allow_pickle=True)

X_test, y_test = load_flow_data('test')
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = [list(classes).index(l) for l in y_test]

labels = list(range(len(classes)))

print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=classes,
    zero_division=0
))

os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(14,12))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Word CNN Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png', dpi=150)
print('Confusion matrix saved to outputs/')