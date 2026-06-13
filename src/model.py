"""MobileNetV2-based gesture classification model."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_gesture_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Transfer-learning model: frozen MobileNetV2 base + trainable classification head.
    Call `unfreeze_top_layers` afterwards for fine-tuning phase.
    """
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="gesture_mobilenetv2")


def unfreeze_top_layers(model: Model, num_layers: int = 30) -> Model:
    """Unfreeze the top `num_layers` of the MobileNetV2 base for fine-tuning."""
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    return model


def compile_model(model: Model, lr: float, num_classes: int) -> Model:
    loss = "categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=["accuracy"],
    )
    return model
