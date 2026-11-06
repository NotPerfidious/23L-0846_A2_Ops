import os
import subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import mlflow
from dotenv import load_dotenv
from src.data_pipeline import data_pipeline
from src.model import build_transfer_model, unfreeze_last_block

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
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

    # ---------------------------------------------------------
    # RUN 1: MobileNetV2 Phase 1 - Base Frozen (Autologging)
    # ---------------------------------------------------------
    print("\n--- Training MobileNetV2 Phase 1 (Autologging) ---")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="MobileNetV2_Phase1_Frozen"):
        mlflow.set_tags({"git_commit": git_commit, "dvc_data_version": "v1.0", "seed": seed_value})
        
        model.fit(train_data, epochs=10, validation_data=val_data)
        
        loss, acc = model.evaluate(test_data, verbose=0)
        mlflow.log_metrics({"final_test_loss": loss, "final_test_accuracy": acc})

    # ---------------------------------------------------------
    # RUN 2: MobileNetV2 Phase 2 - Unfrozen (Manual Logging)
    # ---------------------------------------------------------
    print("\n--- Training MobileNetV2 Phase 2 (Manual Logging) ---")
    mlflow.tensorflow.autolog(disable=True)
    
    # Recompile with smaller learning rate and unfreeze last block
    model = unfreeze_last_block(model, base_model)

    with mlflow.start_run(run_name="MobileNetV2_Phase2_Unfrozen"):
        mlflow.set_tags({"git_commit": git_commit, "dvc_data_version": "v1.0", "seed": seed_value})
        mlflow.log_params({"optimizer": "adam", "learning_rate": 1e-5, "frozen_layers": 143})
        
        model.fit(
            train_data, 
            epochs=10, 
            validation_data=val_data, 
            callbacks=[ManualMLflowLogger()]
        )
        
        loss, acc = model.evaluate(test_data, verbose=0)
        mlflow.log_metrics({"final_test_loss": loss, "final_test_accuracy": acc})

if __name__ == "__main__":
    run_visionguard_training()