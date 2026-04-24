from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from shared.config import (
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    VALIDATION_SPLIT,
    CLASS_NAMES,
)


def load_labels(labels_path):
    """
    Load labels.txt and return a list of class names.
    """
    labels_path = Path(labels_path)
    with labels_path.open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def make_train_val_datasets(
    data_dir,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    seed=SEED,
):
    """
    Create train/validation datasets from a directory structured as:

        data_dir/
            class_1/
            class_2/
            ...

    Returns:
        train_ds, val_ds, class_names
    """
    data_dir = str(Path(data_dir))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # ✅ FIX: Save class_names BEFORE preprocessing
    class_names = train_ds.class_names

    # Now optimize pipeline
    train_ds = prepare_for_training(train_ds, training=True)
    val_ds = prepare_for_training(val_ds, training=False)

    return train_ds, val_ds, class_names

def make_test_dataset(
    test_dir,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
):
    """
    Load a separate test directory if you create one later.
    """
    test_dir = str(Path(test_dir))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return prepare_for_training(test_ds, training=False)


def prepare_for_training(ds, training=False, cache=True):
    """
    Optimized tf.data pipeline.

    - cache: keeps preprocessed batches in memory/disk
    - prefetch: overlaps CPU and GPU work
    """
    if cache:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_augmentation_layer():
    """
    Data augmentation to improve generalization.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def apply_augmentation(ds):
    """
    Apply augmentation batch-wise.
    Use only for the training dataset.
    """
    augmentation = get_augmentation_layer()

    return ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def compute_class_weights(data_dir, class_names=None):
    """
    Compute class weights from folder counts.

    Returns:
        class_weights: dict suitable for model.fit(..., class_weight=class_weights)
        counts: dict with per-class image counts
    """
    data_dir = Path(data_dir)
    if class_names is None:
        class_names = CLASS_NAMES

    counts = {}
    y = []

    for idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        class_count = sum(1 for p in class_dir.iterdir() if p.is_file())
        counts[class_name] = class_count
        y.extend([idx] * class_count)

    y = np.array(y)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
    )

    class_weights = {i: float(w) for i, w in enumerate(class_weight_values)}
    return class_weights, counts


def print_class_counts(counts):
    """
    Print class counts nicely in notebook output.
    """
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")