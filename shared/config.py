from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHARED_DIR = PROJECT_ROOT / "shared"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "PlantVillage"

SPLITS_DIR = DATA_DIR / "splits"
SERVER_TEST_DIR = SPLITS_DIR / "server_test"
PI_CLIENT_DIR = SPLITS_DIR / "pi_client"
ANDROID_CLIENT_DIR = SPLITS_DIR / "android_client"

SERVER_DIR = PROJECT_ROOT / "server"
CLIENTS_DIR = PROJECT_ROOT / "clients"

# --------------------------------------------------
# Model / training settings
# --------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
SEED = 42

EPOCHS = 20
FINE_TUNE_EPOCHS = 10

LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.3
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# --------------------------------------------------
# Class names (must match labels.txt exactly)
# --------------------------------------------------
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------
# Files
# --------------------------------------------------
LABELS_FILE = SHARED_DIR / "labels.txt"
MODEL_NAME = "plant_disease_mobilenetv2"