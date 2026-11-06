import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)
from src.data_pipeline import data_pipeline


def evaluate_models():
    # loading test data
    _, _, test_data = data_pipeline()

    model_paths = {
        "Baseline CNN": "baseline_model.keras",
        "MobileNetV2": "champion_model.keras",
    }

    print("Extracting test data labels and features...")
    y_true = []
    test_images = []

    # Extract labels and images simultaneously to maintain order
    for images, labels in test_data:
        y_true.extend(labels.numpy())
        test_images.append(images)

    y_true = np.array(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    # Evaluate both models
    for model_name, path in model_paths.items():
        print(f"\n--- Evaluating {model_name} ---")
        model = tf.keras.models.load_model(path)

        y_pred_probs = []
        for batch in test_images:
            y_pred_probs.extend(model.predict(batch, verbose=0))

        y_pred_probs = np.array(y_pred_probs).flatten()

        # Apply standard threshold
        threshold = 0.5
        y_pred_classes = (y_pred_probs >= threshold).astype(int)

        # Print metrics for the report table
        print(f"Accuracy:  {accuracy_score(y_true, y_pred_classes):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred_classes):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred_classes):.4f}")
        print(f"F1 Score:  {f1_score(y_true, y_pred_classes):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_pred_probs):.4f}")

        # Generate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Real", "AI"]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.savefig(f"{model_name.replace(' ', '_').lower()}_cm.png")
        plt.close(fig)

        # Add to combined ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("combined_roc_curve.png")



if __name__ == "__main__":
    evaluate_models()
