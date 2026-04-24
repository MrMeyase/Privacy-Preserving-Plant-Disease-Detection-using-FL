from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import CLASS_NAMES
from server.aggregator import GLOBAL_MODEL_PATH

SERVER_TEST_DIR = PROJECT_ROOT / "data" / "splits" / "server_test"

def evaluate_global_model():
    if not GLOBAL_MODEL_PATH.exists():
        return None, None

    model = tf.keras.models.load_model(GLOBAL_MODEL_PATH)
    
    if not SERVER_TEST_DIR.exists():
        return None, None

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(SERVER_TEST_DIR),
        image_size=(224, 224),
        batch_size=32,
        verbose=0 # Keep logs clean
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    return loss, acc

if __name__ == "__main__":
    evaluate_global_model()