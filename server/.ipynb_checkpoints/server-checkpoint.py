import logging
import os
import sys
import struct
import numpy as np
from flask import Flask, request, send_file, jsonify, Response
from pathlib import Path
from threading import Lock

# Setup path to import aggregator
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.aggregator import (
    GLOBAL_MODEL_PATH, TFLITE_MODEL_PATH, load_model_or_create, 
    save_model, rebuild_weights, fedavg, export_to_tflite
)
from server.evaluate import evaluate_global_model

app = Flask(__name__)
lock = Lock()

# Global State
STATE = {
    "round": 1,
    "model": None,
    "pending_updates": []  # List of dicts: {client_id, num_samples, weights}
}

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "checkpoints"
MODEL_DIR.mkdir(exist_ok=True)

@app.route('/health', methods=['GET'])
def health():
    with lock:
        return jsonify({
            "status": "ok", 
            "round": STATE["round"], 
            "pending": len(STATE["pending_updates"])
        }), 200

@app.route('/download_model', methods=['GET'])
@app.route('/get_model', methods=['GET'])
def download_model():
    """Serves the latest TFLite model to the Android app."""
    if TFLITE_MODEL_PATH.exists():
        logging.info(" Sending TFLite model to client")
        return send_file(TFLITE_MODEL_PATH, as_attachment=True)
    return "Model not found", 404

@app.route('/get_global_weights_binary', methods=['GET'])
def get_weights_binary():
    """Converts Keras weights to a flat binary buffer for the phone."""
    with lock:
        weights = STATE["model"].get_weights()
    flat = np.concatenate([w.astype(np.float32).ravel() for w in weights])
    logging.info(f" Sending weights: {len(flat)} parameters")
    return Response(flat.tobytes(), mimetype="application/octet-stream")

@app.route('/upload_weights_binary', methods=['POST'])
def upload_weights_binary():
    """Receives binary weights, reconstructs them, and stores for aggregation."""
    if "weights_file" not in request.files:
        return jsonify({"error": "No file"}), 400

    client_id = request.form.get("client_id", "unknown")
    num_samples = int(request.form.get("num_samples", 1))
    raw_bytes = request.files["weights_file"].read()
    flat_weights = np.frombuffer(raw_bytes, dtype=np.float32)

    with lock:
        template = STATE["model"].get_weights()
        client_weights = rebuild_weights(flat_weights, template)
        STATE["pending_updates"].append({
            "client_id": client_id,
            "num_samples": num_samples,
            "weights": client_weights
        })
        logging.info(f" Received weights from {client_id} ({num_samples} samples). Pending: {len(STATE['pending_updates'])}")

    return jsonify({"status": "received", "pending": len(STATE["pending_updates"])}), 200

@app.route('/upload_prediction', methods=['POST'])
def upload_prediction():
    data = request.get_json()
    label = data.get("label", "Unknown")
    confidence = data.get("confidence", 0.0)
    logging.info(f" PREDICTION: {label} | Confidence: {confidence:.2f}")
    return jsonify({"status": "ok"}), 200

@app.route("/evaluate", methods=["GET"])
def evaluate():
    logging.info(f" Evaluating Global Model...")
    with lock:
        loss, acc = evaluate_global_model()
    
    if loss is None:
        return jsonify({"error": "Evaluation failed or no test data"}), 500
        
    return jsonify({
        "status": "success",
        "loss": float(loss),
        "accuracy": float(acc),
        "round": STATE["round"]
    }), 200

@app.route("/aggregate", methods=["POST"])
def aggregate():
    with lock:
        if not STATE["pending_updates"]:
            logging.warning(" No updates found in pending_updates.")
            return jsonify({"error": "No updates to aggregate"}), 400
        
        logging.info(f" AGGREGATING {len(STATE['pending_updates'])} updates...")
        weights_list = [u["weights"] for u in STATE["pending_updates"]]
        samples_list = [u["num_samples"] for u in STATE["pending_updates"]]
        
        new_weights = fedavg(weights_list, samples_list)
        STATE["model"].set_weights(new_weights)
        save_model(STATE["model"], GLOBAL_MODEL_PATH)
        export_to_tflite(STATE["model"], TFLITE_MODEL_PATH)
        
        num_clients = len(STATE["pending_updates"])
        STATE["pending_updates"].clear()
        STATE["round"] += 1
        logging.info(f" ROUND {STATE['round']-1} COMPLETE.")
    
    return jsonify({"status": "aggregated", "new_round": STATE["round"], "clients": num_clients}), 200

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    STATE["model"] = load_model_or_create()
    app.run(host="0.0.0.0", port=8000, threaded=True)