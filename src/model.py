from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

def build_model(num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=output)
