import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import mlflow
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def build_and_train(learning_rate, run_name):

    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:1000] / 255.0  # Subset for speed
    y_train = y_train[:1000]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("learning_rate", learning_rate)

        history = model.fit(x_train, y_train, epochs=5, verbose=0)

        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric(
                "accuracy", history.history["accuracy"][epoch], step=epoch
            )

        plt.figure()
        plt.plot(history.history["loss"], label="Loss")
        plt.title(f"Training Loss (LR={learning_rate})")
        plt.legend()
        plot_path = f"loss_plot_{run_name}.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        mlflow.tensorflow.log_model(model, "toy_model")

        print(f"Finished Run: {run_name} (LR: {learning_rate})")


if __name__ == "__main__":

    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    # print(f"\nTracking uri: {tracking_uri}\n")

    # mlflow.set_experiment("Toy_fashion_MNIST")

    mlflow.set_experiment("Toy_fashion_MNIST_Remote")

    build_and_train(learning_rate=0.01, run_name="Run_High_LR")

    build_and_train(learning_rate=0.001, run_name="Run_Low_LR")
