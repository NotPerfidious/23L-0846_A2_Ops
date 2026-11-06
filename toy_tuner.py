import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
import keras_tuner as kt


def build_toy_model(hp):
    # Search space: units and learning rate
    hp_units = hp.Int("units", min_value=32, max_value=128, step=32)
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hp_units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_toy_tuner():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_val = x_train / 255.0, x_val / 255.0

    tuner = kt.RandomSearch(
        build_toy_model,
        objective="val_accuracy",
        max_trials=5,
        directory="toy_tuning",
        project_name="fashion_mnist",
    )

    print("--- Starting Toy Tuner (Fashion-MNIST) ---")
    tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest Toy Units: {best_hps.get('units')}")
    print(f"Best Toy LR: {best_hps.get('learning_rate')}")

   
    print("--- Starting Toy Tuner (Fashion-MNIST) ---")
    tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val))

   
    trials = tuner.oracle.get_best_trials(num_trials=5)
    toy_data = []
    for t in trials:
        row = t.hyperparameters.values.copy()
        row["val_accuracy"] = t.score
        toy_data.append(row)

    df_toy = pd.DataFrame(toy_data)
    print("\n" + "=" * 40)
    print("TOY RESULTS TABLE")
    print("=" * 40)
    print(df_toy.to_string(index=False))


if __name__ == "__main__":
    run_toy_tuner()
