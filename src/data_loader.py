import os
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input

def load_data_paths(base_path):
    classes = ['Mild Dementia','Moderate Dementia','Non Demented','Very mild Dementia']
    data_dirs = [os.path.join(base_path, cls) for cls in classes]
    all_images, all_labels = [], []

    for idx, dir_path in enumerate(data_dirs):
        full_dir = pathlib.Path(dir_path)
        images = list(full_dir.glob('*'))
        all_images.extend([str(img) for img in images])
        all_labels.extend([idx]*len(images))

    return np.array(all_images), np.array(all_labels), classes

def preprocess(image, label, num_classes):
    img = tf.io.read_file(image)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224,224])
    img = preprocess_input(img)
    return img, tf.one_hot(label, depth=num_classes)

def get_datasets(base_path, batch_size=128):
    all_images, all_labels, classes = load_data_paths(base_path)

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, len(classes)), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, len(classes)), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(lambda x, y: preprocess(x, y, len(classes)), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, classes
