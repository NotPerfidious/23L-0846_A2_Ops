import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.data_pipeline import data_pipeline

def generate_error_grid():
    _, _, test_data = data_pipeline()
    model = tf.keras.models.load_model('champion_model.keras')

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    print("Hunting for misclassified images...")
    
    # Scan through batches until we find 10 errors
    for images, labels in test_data:
        preds = model.predict(images, verbose=0).flatten()
        pred_classes = (preds >= 0.5).astype(int)
        true_classes = labels.numpy()

        for i in range(len(true_classes)):
            if pred_classes[i] != true_classes[i]:
                misclassified_images.append(images[i].numpy().astype("uint8"))
                misclassified_labels.append(true_classes[i])
                misclassified_preds.append(preds[i])
                
                if len(misclassified_images) == 10:
                    break
        if len(misclassified_images) == 10:
            break

    # Plot the 2x5 grid
    class_names = {0: "Real", 1: "AI"}
    plt.figure(figsize=(15, 6))
    
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(misclassified_images[i])
        
        true_label = class_names[misclassified_labels[i]]
        pred_label = class_names[1 - misclassified_labels[i]] # If it's wrong, it guessed the opposite
        confidence = misclassified_preds[i] if pred_label == "AI" else 1 - misclassified_preds[i]
        
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("error_analysis_grid.png")
    print("Saved error_analysis_grid.png to disk.")

if __name__ == "__main__":
    generate_error_grid()