import os
from pathlib import Path
from PIL import Image
import random

# ---------------- CONFIG ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESIZED_DIR = PROCESSED_DIR / "resized"
AUGMENTED_DIR = PROCESSED_DIR / "augmented"

IMG_SIZE = (224, 224)
AUGMENT_PER_IMAGE = 2   # number of augmented images per original

SEED = 42
random.seed(SEED)
# ---------------------------------------- #


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if path.exists():
        import shutil
        shutil.rmtree(path)
    os.makedirs(path)


# ---------------- RESIZE ---------------- #
def resize_and_save(image_path, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(save_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# ---------------- AUGMENT ---------------- #
def augment_image(img):
    ops = []

    # Flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)

    # Slight brightness change
    if random.random() > 0.5:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img


def process_class(class_dir):
    class_name = class_dir.name

    resized_class_dir = RESIZED_DIR / class_name
    augmented_class_dir = AUGMENTED_DIR / class_name

    create_dir(resized_class_dir)
    create_dir(augmented_class_dir)

    images = [p for p in class_dir.iterdir() if p.is_file()]

    print(f"\n📂 Processing {class_name} ({len(images)} images)")

    for img_path in images:
        # -------- Resize -------- #
        resized_path = resized_class_dir / img_path.name
        resize_and_save(img_path, resized_path)

        # -------- Augment -------- #
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            for i in range(AUGMENT_PER_IMAGE):
                aug_img = augment_image(img)

                aug_name = f"{img_path.stem}_aug_{i}.jpg"
                aug_path = augmented_class_dir / aug_name

                aug_img.save(aug_path)

        except Exception as e:
            print(f"Aug error {img_path}: {e}")


def main():
    print("🚀 Starting preprocessing...")

    clear_dir(RESIZED_DIR)
    clear_dir(AUGMENTED_DIR)

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        process_class(class_dir)

    print("\n✅ Preprocessing completed!")


if __name__ == "__main__":
    main()