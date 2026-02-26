import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score

class AdaptiveAlphaGammaCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, loss_fn, class_names):
        super().__init__()
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for x_batch, y_batch in self.val_data:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))

        recalls = recall_score(y_true, y_pred, labels=list(range(len(self.class_names))), average=None)

        min_beta, max_beta = 0.01, 0.9
        beta_dynamics = min_beta + (max_beta - min_beta) * recalls
        new_gamma = 2 + (1 - recalls) * 2
        new_alpha = 1 + (1 - recalls) * 2

        updated_gamma = beta_dynamics * self.loss_fn.gamma + (1 - beta_dynamics) * new_gamma
        updated_alpha = beta_dynamics * self.loss_fn.alpha + (1 - beta_dynamics) * new_alpha

        self.loss_fn.gamma.assign(tf.convert_to_tensor(updated_gamma, dtype=tf.float32))
        self.loss_fn.alpha.assign(tf.convert_to_tensor(updated_alpha, dtype=tf.float32))

        print(f"[AdaptiveAlphaGamma] Updated gamma: {updated_gamma}")
        print(f"[AdaptiveAlphaGamma] Updated alpha: {updated_alpha}")
