import os
import numpy as np
import mlflow
from dotenv import load_dotenv
from src.data_pipeline import resized_image
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

MODEL_PATH = os.getenv("MODEL_PATH", "models:/VisionGuard_Classifier@champion")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

print(f"[START-UP] Fetching Model from {MODEL_PATH}...")
_CHAMPION_MODEL = mlflow.tensorflow.load_model(MODEL_PATH)
print("[START-UP] Model loaded successfully and ready for fast inference.\n")


def predict_image(image_path: str):
    """
    Preprocesses an uploaded image exactly as in training,
    and returns a label and a confidence.
    """

    img_tensor = resized_image(image_path)
    img_batch = np.expand_dims(img_tensor, axis=0)

    prediction = _CHAMPION_MODEL.predict(img_batch, verbose=0)
    score = float(prediction[0][0])

    if score > 0.5:
        label = "AI Art"
        confidence = score
    else:
        label = "Real Art"
        confidence = 1.0 - score  # Convert to standard percentage format

    return {"label": label, "confidence": round(confidence * 100, 2)}


def predict_image_bytes(image_bytes):
    """
    Preprocesses raw image bytes and predicts using the configured threshold.
    """
    try:
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img_tensor = tf.image.resize_with_pad(img, target_height=224, target_width=224)
        img_batch = np.expand_dims(img_tensor, axis=0)

        prediction = _CHAMPION_MODEL.predict(img_batch, verbose=0)
        score = float(prediction[0][0])

        # 2. APPLY ENVIRONMENT THRESHOLD
        if score > THRESHOLD:
            label = "AI Art"
            confidence = score
        else:
            label = "Real Art"
            confidence = 1.0 - score

        return {"label": label, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        raise ValueError("The file looks like an image but cannot be decoded.")


if __name__ == "__main__":

    real_img = "data/raw/RealArt/pexels-photo-2422290.jpeg"

    ai_img = "data/raw/AiArtData/Midjourney-AI-Art-Clown.png"

    print(f"Testing Real Image ({real_img}):")
    print(predict_image(real_img))
    print("-" * 40)

    print(f"Testing AI Image ({ai_img}):")
    print(predict_image(ai_img))
