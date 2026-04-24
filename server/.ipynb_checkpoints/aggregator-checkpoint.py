from __future__ import annotations
import io
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import tensorflow as tf

# Setup path to shared module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import NUM_CLASSES
from shared.model import build_model, compile_model

# --- Directory Definitions ---
SERVER_DIR = PROJECT_ROOT / "server"
CHECKPOINTS_DIR = SERVER_DIR / "checkpoints"
ROUNDS_DIR = SERVER_DIR / "rounds"
LOGS_DIR = SERVER_DIR / "logs"

# --- Model Paths ---
GLOBAL_MODEL_PATH = CHECKPOINTS_DIR / "global_model.keras"
GLOBAL_WEIGHTS_PATH = CHECKPOINTS_DIR / "global_weights.npz"
TFLITE_MODEL_PATH = CHECKPOINTS_DIR / "global_model.tflite"

def rebuild_weights(flat_weights: np.ndarray, template_weights: List[np.ndarray]) -> List[np.ndarray]:
    """Converts a flat float32 array back into a list of correctly shaped numpy arrays."""
    rebuilt = []
    idx = 0
    for w in template_weights:
        size = int(np.prod(w.shape))
        rebuilt.append(flat_weights[idx : idx + size].reshape(w.shape))
        idx += size
    return rebuilt

def fedavg(client_weights_list: List[List[np.ndarray]], sample_counts: Sequence[int]) -> List[np.ndarray]:
    """Performs Federated Averaging (FedAvg)."""
    if not client_weights_list:
        raise ValueError("No weights to aggregate")
    
    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count is zero")

    aggregated = [np.zeros_like(w, dtype=np.float32) for w in client_weights_list[0]]
    
    for client_weights, count in zip(client_weights_list, sample_counts):
        weight_factor = count / total_samples
        for i in range(len(aggregated)):
            aggregated[i] += client_weights[i] * weight_factor
            
    return aggregated

def load_model_or_create(path: Path = GLOBAL_MODEL_PATH) -> tf.keras.Model:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return tf.keras.models.load_model(path)
    
    logging.info("Creating new global model...")
    model = build_model(num_classes=NUM_CLASSES)
    model = compile_model(model)
    model.save(path)
    return model

def save_model(model: tf.keras.Model, path: Path = GLOBAL_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

def export_to_tflite(model: tf.keras.Model, path: Path = TFLITE_MODEL_PATH) -> None:
    """Converts the Keras model to TFLite format for the Android app."""
    logging.info(f"Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    logging.info(f"TFLite model saved to {path}")