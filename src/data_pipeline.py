import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from tensorflow._api.v2 import io as tf_io
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow._api.v2 import image as tf_img
from tensorflow._api.v2 import data as tf_data
import matplotlib.pyplot as plt

ai_art_data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "AiArtData"
real_art_data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "RealArt"

BATCH_SIZE = 32


def split_lists():
    X = []
    y = []  # 0:Real art 1:AI art

    for item in real_art_data_path.iterdir():
        X.append(str(item))
        y.append(0)

    for item in ai_art_data_path.iterdir():
        X.append(str(item))
        y.append(1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def resized_image(image_path):

    content = tf_io.read_file(image_path)
    img = tf_io.decode_image(content, channels=3, expand_animations=False)

    return tf_img.resize_with_pad(img, target_height=224, target_width=224)


def apply_random_augmentation(dataset):

    augmentor = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomRotation(0.05),
        ]
    )

    dataset = dataset.map(
        lambda images, labels: (augmentor(images, training=True), labels)
    )

    dataset_iterator = iter(dataset)

    images, labels = next(dataset_iterator)

    plt.figure(figsize=(12, 8))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        if labels[i] == 0:
            plt.title("Real art")
        else:
            plt.title("AI art")
        plt.axis("off")

    plt.savefig("augmented_batch.png")

    return dataset


def data_pipeline_worker(X: list, y: list, is_training=False):

    dataset = tf_data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.map(lambda image, label: (resized_image(image), label))

    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

    if is_training:
        dataset = apply_random_augmentation(dataset)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def data_pipeline():

    X_train, y_train, X_val, y_val, X_test, y_test = split_lists()

    train_data = data_pipeline_worker(X_train, y_train, is_training=True)
    val_data = data_pipeline_worker(X_val, y_val)
    test_data = data_pipeline_worker(X_test, y_test)

    return train_data, val_data, test_data


if __name__ == "__main__":

    data_pipeline()
