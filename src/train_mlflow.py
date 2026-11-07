import os
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import mlflow
from dotenv import load_dotenv
from src.data_pipeline import data_pipeline
from src.model import build_transfer_model, unfreeze_last_block
from src.data_pipeline import split_lists
import csv


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown_commit"


class ManualMLflowLogger(tf.keras.callbacks.Callback):
    """Custom callback for manual per-epoch metric logging."""

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for metric, value in logs.items():
                mlflow.log_metric(metric, value, step=epoch)


def run_visionguard_training():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("VisionGuard_Actual_Training")

    train_data, val_data, test_data = data_pipeline()
    git_commit = get_git_commit()
    seed_value = 42
    tf.keras.utils.set_random_seed(seed_value)

    # Build and compile the base model for Phase 1
    model, base_model = build_transfer_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # MobileNetV2 Phase 1 - Base Frozen (Autologging)
    print("\n------------------------ Training MobileNetV2 Phase 1 (Autologging)")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="MobileNetV2_Phase1_Frozen"):
        mlflow.set_tags(
            {"git_commit": git_commit, "dvc_data_version": "v1.0", "seed": seed_value}
        )

        model.fit(train_data, epochs=10, validation_data=val_data)

        loss, acc = model.evaluate(test_data, verbose=0)
        mlflow.log_metrics({"final_test_loss": loss, "final_test_accuracy": acc})
        
        # Explicitly register Phase 1 as Version 1
        print("Registering Phase 1 model...")
        mlflow.tensorflow.log_model(
            model, 
            "model_phase1", 
            registered_model_name="VisionGuard_Classifier"
        )

    # MobileNetV2 Phase 2 - Unfrozen (Manual Logging)
    print("\n------------------------ Training MobileNetV2 Phase 2 (Manual Logging)")
    mlflow.tensorflow.autolog(disable=True)

    # Recompile with smaller learning rate and unfreeze last block
    model = unfreeze_last_block(model, base_model)

    with mlflow.start_run(run_name="MobileNetV2_Phase2_Unfrozen"):
        mlflow.set_tags(
            {"git_commit": git_commit, "dvc_data_version": "v1.0", "seed": seed_value}
        )
        mlflow.log_params(
            {"optimizer": "adam", "learning_rate": 1e-5, "frozen_layers": 143}
        )

        model.fit(
            train_data,
            epochs=10,
            validation_data=val_data,
            callbacks=[ManualMLflowLogger()],
        )

        loss, acc = model.evaluate(test_data, verbose=0)
        mlflow.log_metrics({"final_test_loss": loss, "final_test_accuracy": acc})

        print("Saving and Uploading Artifacts to DagsHub...")

        X_train, y_train, X_val, y_val, X_test, y_test = split_lists()

        def save_split(filename, X, y):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filepath", "label"])
                writer.writerows(zip(X, y))

        save_split("train_split.csv", X_train, y_train)
        save_split("val_split.csv", X_val, y_val)
        save_split("test_split.csv", X_test, y_test)

       
        mlflow.log_artifact("train_split.csv")
        mlflow.log_artifact("val_split.csv")
        mlflow.log_artifact("test_split.csv")

       
        mlflow.log_artifact("mobilenetv2_cm.png")
        mlflow.log_artifact("combined_roc_curve.png")

        
        mlflow.log_artifact("requirements.txt")

       
        model.save("champion_visionguard.keras")
        mlflow.log_artifact("champion_visionguard.keras")
        
       
        print("Registering Phase 2 Champion model...")
        mlflow.tensorflow.log_model(
            model, 
            "model_phase2", 
            registered_model_name="VisionGuard_Classifier"
        )

        print("Artifacts successfully uploaded and models registered!")


if __name__ == "__main__":
    run_visionguard_training()