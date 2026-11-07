import os
import gc
import tensorflow as tf
import mlflow
from dotenv import load_dotenv
from src.data_pipeline import data_pipeline
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_last_block

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_experiment(run_name, backbone, learning_rate, train_data, val_data, frozen=True):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "backbone": backbone,
            "learning_rate": learning_rate,
            "frozen_layers": "All" if frozen else "Base-Only"
        })
        
        if backbone == "Baseline":
            model = build_baseline_cnn()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy', metrics=['accuracy'])
        elif backbone == "MobileNetV2":
            model, base_model = build_transfer_model()
            if not frozen:
                model = unfreeze_last_block(model, base_model)
                
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy', metrics=['accuracy'])
        
        print(f"--- Running: {run_name} ---")
        history = model.fit(train_data, epochs=3, validation_data=val_data, verbose=1)
        
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metrics({"val_loss": final_val_loss, "val_accuracy": final_val_acc})

    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("VisionGuard_Comparison_Task3.5")
    
    print("Loading dataset into memory ONCE...")
    train_data, val_data, _ = data_pipeline()
    
    configs = [
        ("Run_1_Baseline", "Baseline", 1e-3, True),
        ("Run_2_Baseline", "Baseline", 5e-4, True),
        ("Run_3_Baseline", "Baseline", 1e-4, True),
        ("Run_4_MobileNet", "MobileNetV2", 1e-3, True),
        ("Run_5_MobileNet", "MobileNetV2", 1e-4, True),
        ("Run_6_MobileNet_Unfrozen", "MobileNetV2", 1e-5, False)
    ]
    
    for name, b_bone, lr, is_frozen in configs:
        run_experiment(name, b_bone, lr, train_data, val_data, is_frozen)
        