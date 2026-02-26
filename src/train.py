from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from src.data_loader import get_datasets
from src.model import build_model
from src.losses import categorical_focal_loss, AdaptiveCategoricalFocalLoss
from src.callbacks import AdaptiveAlphaGammaCallback

def main():
    base_path = '/kaggle/input/imagesoasis/Data'
    batch_size = 128
    initial_lr = 0.001
    fine_tune_lr = 0.0001

    train_ds, val_ds, test_ds, classes = get_datasets(base_path, batch_size)

    model = build_model(len(classes))

    alpha = [4.32055127, 44.22880117, 0.32145893, 1.57450297]
    gamma = [2, 2, 2, 2]

    # Phase 1 training
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss=categorical_focal_loss(alpha, gamma), metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(monitor='val_loss', filepath='best_model.keras', save_best_only=True, verbose=1)

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stopping, checkpoint])

    # Fine tuning
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    optimizer = Adam(learning_rate=fine_tune_lr)
    loss_fn = AdaptiveCategoricalFocalLoss(alpha, gamma)
    alpha_gamma_cb = AdaptiveAlphaGammaCallback(val_ds, loss_fn, classes)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(monitor='val_loss', filepath='fine_tune_best_model.keras', save_best_only=True, verbose=1)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stopping, checkpoint, reduce_lr, alpha_gamma_cb])

if __name__ == "__main__":
    main()
