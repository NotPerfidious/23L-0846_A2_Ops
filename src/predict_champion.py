import os
import numpy as np
import mlflow
from dotenv import load_dotenv
from src.data_pipeline import resized_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_inference():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model_name = "VisionGuard_Classifier"
    alias = "champion"

    model_uri = f"models:/{model_name}@{alias}"

    print(f"Downloading '{alias}' model from DagsHub Registry ({model_uri})...")
    champion_model = mlflow.tensorflow.load_model(model_uri)
    print("Model loaded successfully!\n")

    test_image = "data/raw/RealArt/12_20201217171040_10852275_large.jpg"

    print(f"Classifying image: {test_image}")

    img_tensor = resized_image(test_image)

    img_batch = np.expand_dims(img_tensor, axis=0)

    raw_pred = champion_model.predict(img_batch, verbose=0)
    score = raw_pred[0][0]

    label = "AI Art (Class 1)" if score > 0.5 else "Real Art (Class 0)"

    print("\n-----------------------------------------------------")
    print(f"Prediction Score : {score:.4f}")
    print(f"Final Class      : {label}")


if __name__ == "__main__":
    run_inference()
