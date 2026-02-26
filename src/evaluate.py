import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data_loader import get_datasets

def evaluate():
    base_path = '/kaggle/input/imagesoasis/Data'
    _, _, test_ds, classes = get_datasets(base_path)

    model = tf.keras.models.load_model('fine_tune_best_model.keras', compile=False)

    y_pred, y_true = [], []

    for images, labels in test_ds:
        pred_prob = model.predict(images)
        y_pred.extend(np.argmax(pred_prob, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()

if __name__ == "__main__":
    evaluate()
