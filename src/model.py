import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.data_pipeline import data_pipeline
import matplotlib.pyplot as plt


def build_baseline_cnn():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            #
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            #
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            #
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            #
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            #
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def build_transfer_model(input_shape=(224, 224, 3)):

    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = preprocess_input(inputs)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def unfreeze_last_block(model, base_model):
    # Unfreeze the base model completely
    base_model.trainable = True

    # MobileNetV2 has ~154 layers. Layers from 143 onwards form the final block.
    for layer in base_model.layers[:143]:
        layer.trainable = False

    # Using a drastically smaller learning rate to avoid destroying pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    train_data, val_data, _ = data_pipeline()

    pretained_model, base_model = build_transfer_model()

    pretained_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    print("Trainable layers (Base frozen):")
    for layer in pretained_model.layers:
        if layer.trainable:
            print(layer.name)

    history1 = pretained_model.fit(train_data, epochs=10, validation_data=val_data)
    train_loss1 = history1.history["loss"]
    val_loss1 = history1.history["val_loss"]
    train_acc1 = history1.history["accuracy"]
    val_acc1 = history1.history["val_accuracy"]

    pretained_model = unfreeze_last_block(pretained_model, base_model)

    print("\nTrainable layers (Last Block unfrozen):")
    for layer in pretained_model.layers:
        if layer.trainable:
            print(layer.name)

    history2 = pretained_model.fit(train_data, epochs=10, validation_data=val_data)
    train_loss2 = history2.history["loss"]
    val_loss2 = history2.history["val_loss"]
    train_acc2 = history2.history["accuracy"]
    val_acc2 = history2.history["val_accuracy"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss1, label="Training Loss (Base Frozen)")
    plt.plot(val_loss1, label="Validation Loss (Base Frozen)")
    plt.plot(train_loss2, label="Training Loss (Last Block unfrozen)")
    plt.plot(val_loss2, label="Validation Loss (Last Block unfrozen)")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc1, label="Training Accuracy (Base Frozen)")
    plt.plot(val_acc1, label="Validation Accuracy (Base Frozen)")
    plt.plot(train_acc2, label="Training Accuracy (Last Block unfrozen)")
    plt.plot(val_acc2, label="Validation Accuracy (Last Block unfrozen)")
    plt.title("Training vs Validation Accuuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("base_frozen_vs_last_block_unfrozen.png")
