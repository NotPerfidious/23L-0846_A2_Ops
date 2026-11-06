import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.data_pipeline import data_pipeline
import pandas as pd


def build_visionguard_tuner_model(hp):
    # 1. Hyperparameter Search Space
    hp_lr = hp.Choice("learning_rate", values=[1e-4, 5e-5, 1e-5])
    hp_dropout = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    hp_unfrozen = hp.Int("unfrozen_layers", min_value=10, max_value=30, step=10)

    # 2. Base Architecture
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    # Unfreeze only the last `hp_unfrozen` layers
    base_model.trainable = True
    for layer in base_model.layers[:-hp_unfrozen]:
        layer.trainable = False

    # 3. Custom Head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_tuning():
    train_data, val_data, _ = data_pipeline()

    tuner = kt.RandomSearch(
        build_visionguard_tuner_model,
        objective="val_loss",
        max_trials=8,
        executions_per_trial=1,
        directory="tuning_dir",
        project_name="visionguard_actual",
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    print("\n--- Starting VisionGuard Tuner ---")
    tuner.search(train_data, epochs=8, validation_data=val_data, callbacks=[stop_early])

    # Generate the VisionGuard Results Table
    trials = tuner.oracle.get_best_trials(num_trials=8)
    vg_data = []
    for t in trials:
        row = t.hyperparameters.values.copy()
        row["val_loss"] = t.score
        vg_data.append(row)

    df_vg = pd.DataFrame(vg_data)
    print("\n" + "=" * 50)
    print("VISIONGUARD TUNER RESULTS TABLE")
    print("=" * 50)
    print(df_vg.to_string(index=False))


if __name__ == "__main__":
    run_tuning()
