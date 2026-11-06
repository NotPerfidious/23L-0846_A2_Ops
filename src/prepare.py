from pathlib import Path
from PIL import Image, UnidentifiedImageError
import hashlib
import pandas as pd
import random
import matplotlib.pyplot as plt


def get_image_hash(file_path: Path):

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:

        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def get_random_images(folder_path: Path, n: int):

    seen_hashes = set()
    images = []

    for item in folder_path.iterdir():
        if item.is_file():

            img_hash = get_image_hash(item)

            if img_hash not in seen_hashes:

                seen_hashes.add(img_hash)
                images.append(item)

    if n > len(images):
        n = len(images)

    return random.sample(images, n)


def read_image_files(folder_path: Path, class_name: str, raw_images: dict):

    corrupted_files = 0
    duplicate_files = 0

    seen_hashes = set(raw_images["content-hash"])

    for item in folder_path.iterdir():
        if item.is_file():
            try:

                with Image.open(item) as img:

                    img.load()  # verifies that image file is not correupted

                    img_hash = get_image_hash(item)

                    if img_hash not in seen_hashes:
                        raw_images["class"].append(class_name)
                        raw_images["file-size"].append(item.stat().st_size)
                        raw_images["width"].append(img.width)
                        raw_images["height"].append(img.height)
                        raw_images["colour-mode"].append(img.mode)
                        raw_images["format"].append(img.format)
                        raw_images["content-hash"].append(img_hash)

                        seen_hashes.add(img_hash)

                    else:
                        duplicate_files += 1

            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                # the file is corrupted
                corrupted_files += 1

            except Exception as e:
                pass

    return corrupted_files, duplicate_files


ai_art_data_path = Path("../data/raw/AiArtData")
real_art_data_path = Path("../data/raw/RealArt")

raw_data = {
    "class": [],
    "file-size": [],
    "width": [],
    "height": [],
    "colour-mode": [],
    "format": [],
    "content-hash": [],
}

corrupted_files, duplicate_files = read_image_files(
    ai_art_data_path, "AI-generated", raw_data
)
c, d = read_image_files(real_art_data_path, "Real", raw_data)

corrupted_files += c
duplicate_files += d

print()

print(f"Total images (AI-generated + Real): {len(raw_data['class'])}")

print()

df = pd.DataFrame(raw_data)

print(df["class"].value_counts())

print()

print(df[["width", "height", "file-size"]].agg(["min", "median", "max"]))

print()

print(f"Number of corrupted files: {corrupted_files}")
print(f"Number of duplicate files: {duplicate_files}")

print()


# ------------------------------------------------ Task 1.3


random_ai_images = get_random_images(ai_art_data_path, 3)
random_real_images = get_random_images(real_art_data_path, 3)

random_images = random_ai_images + random_real_images

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

for i, ax in enumerate(axes.flatten()):

    ax.imshow(Image.open(random_images[i]))

    title = "AI image"

    if i >= 3:
        title = "Real Image"

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

ai_df = df[df["class"] == "AI-generated"]
real_df = df[df["class"] == "Real"]

plt.figure(figsize=(12, 4))

plt.subplot(1,3,1)
plt.hist(ai_df['width'],alpha=0.5, bins=50, label="AI")
plt.hist(real_df['width'], alpha=0.5, bins=50, label="Real")
plt.title("Width")
plt.legend()

plt.subplot(1,3,2)
plt.hist(ai_df['height'],alpha=0.5, bins=50, label="AI")
plt.hist(real_df['height'], alpha=0.5, bins=50, label="Real")
plt.title("Height")
plt.legend()

plt.subplot(1,3,3)
plt.hist(ai_df['file-size'],alpha=0.5, bins=50, label="AI")
plt.hist(real_df['file-size'], alpha=0.5, bins=50, label="Real")
plt.title("File size")
plt.legend()

plt.tight_layout()
plt.show()
