import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.data_pipeline import data_pipeline
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_last_block

def get_callbacks(filepath):
    """Generates the three required callbacks for training discipline."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, 
        monitor='val_loss', 
        save_best_only=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2
    )
    return [early_stopping, checkpoint, reduce_lr]

def plot_curves(train_loss, val_loss, train_acc, val_acc, best_epoch, filename, title):
    """Plots and saves the learning curves with the best epoch highlighted."""
    plt.figure(figsize=(12, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')
    plt.title(f"{title} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()

def train_models():
    # 1. Fix random seeds for reproducibility
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    train_data, val_data, _ = data_pipeline()


    # Train Baseline CNN
    print("\n--- Training Baseline CNN ---")
    baseline_model = build_baseline_cnn()
    baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    baseline_history = baseline_model.fit(
        train_data, 
        epochs=15, 
        validation_data=val_data,
        callbacks=get_callbacks('baseline_model.keras')
    )

    baseline_val_loss = baseline_history.history["val_loss"]
    baseline_best_epoch = np.argmin(baseline_val_loss)
    print(f"Baseline CNN Best Epoch: {baseline_best_epoch + 1}")
    
    plot_curves(
        baseline_history.history["loss"], baseline_val_loss,
        baseline_history.history["accuracy"], baseline_history.history["val_accuracy"],
        baseline_best_epoch, "baseline_training_curves.png", "Baseline CNN"
    )

 
    # Train MobileNetV2 (Transfer Learning)
    print("\n--- Training MobileNetV2 ---")
    transfer_model, base_model = build_transfer_model()
    transfer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    mobilenet_callbacks = get_callbacks('champion_model.keras')

    print("Phase 1: Base Frozen")
    history1 = transfer_model.fit(
        train_data, epochs=10, validation_data=val_data, callbacks=mobilenet_callbacks
    )

    print("\nPhase 2: Last Block Unfrozen")
    transfer_model = unfreeze_last_block(transfer_model, base_model)
    history2 = transfer_model.fit(
        train_data, epochs=10, validation_data=val_data, callbacks=mobilenet_callbacks
    )

    # Combine histories to find the absolute best epoch across both phases
    val_loss_total = history1.history["val_loss"] + history2.history["val_loss"]
    train_loss_total = history1.history["loss"] + history2.history["loss"]
    val_acc_total = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    train_acc_total = history1.history["accuracy"] + history2.history["accuracy"]

    transfer_best_epoch = np.argmin(val_loss_total)
    print(f"MobileNetV2 Best Epoch: {transfer_best_epoch + 1}")

    plot_curves(
        train_loss_total, val_loss_total,
        train_acc_total, val_acc_total,
        transfer_best_epoch, "mobilenet_training_curves.png", "MobileNetV2"
    )

if __name__ == "__main__":
    train_models()