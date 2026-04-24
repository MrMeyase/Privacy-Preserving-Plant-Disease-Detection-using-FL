import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SERVER_TEST_DIR = SPLIT_DIR / "server_test"
PI_CLIENT_DIR = SPLIT_DIR / "pi_client"
ANDROID_CLIENT_DIR = SPLIT_DIR / "android_client"

# Split ratios (must sum to 1)
SERVER_RATIO = 0.15
PI_RATIO = 0.425
ANDROID_RATIO = 0.425

SEED = 42
random.seed(SEED)

# ---------------------------------------- #


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path)


def split_class_images(class_dir):
    images = [p for p in class_dir.iterdir() if p.is_file()]
    random.shuffle(images)

    n_total = len(images)

    n_server = int(n_total * SERVER_RATIO)
    n_pi = int(n_total * PI_RATIO)

    server_imgs = images[:n_server]
    pi_imgs = images[n_server:n_server + n_pi]
    android_imgs = images[n_server + n_pi:]

    return server_imgs, pi_imgs, android_imgs


def copy_images(images, dest_dir):
    create_dir(dest_dir)
    for img in images:
        shutil.copy(img, dest_dir / img.name)


def main():
    print("🚀 Starting dataset split...")

    # Reset split folders
    clear_dir(SERVER_TEST_DIR)
    clear_dir(PI_CLIENT_DIR)
    clear_dir(ANDROID_CLIENT_DIR)

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n📂 Processing class: {class_name}")

        server_imgs, pi_imgs, android_imgs = split_class_images(class_dir)

        # Destination folders
        server_dest = SERVER_TEST_DIR / class_name
        pi_dest = PI_CLIENT_DIR / class_name
        android_dest = ANDROID_CLIENT_DIR / class_name

        print(f"   Server: {len(server_imgs)}")
        print(f"   Pi: {len(pi_imgs)}")
        print(f"   Android: {len(android_imgs)}")

        copy_images(server_imgs, server_dest)
        copy_images(pi_imgs, pi_dest)
        copy_images(android_imgs, android_dest)

    print("\n✅ Dataset split completed successfully!")


if __name__ == "__main__":
    main()