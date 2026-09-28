import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "edge_vision" / "web" / "static" / "uploads"
STATIC_DIR = BASE_DIR / "edge_vision" / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "edge_vision" / "web" / "templates"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'edge_vision.db'}")

# Machine Learning & CV Models
YOLO_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
CNN_MODEL_PATH = str(MODELS_DIR / "best_fruit_quality_model.pth")
CLASS_NAMES_PATH = str(MODELS_DIR / "class_names.json")

# Intelligence & LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Spoilage & Freshness Thresholds
CRITICAL_FRESHNESS_PERCENT = 30.0  # Items < 30% trigger recipes & alerts
CRITICAL_DAYS_LEFT = 2            # Items <= 2 days trigger recipes & alerts
SPOILAGE_FRESHNESS_PERCENT = 70.0 # Items between 30% and 70% are APPROACHING_SPOILAGE

# Ambient Alerts Webhook (WhatsApp / Webhook Simulator)
AMBIENT_WEBHOOK_URL = os.getenv("AMBIENT_WEBHOOK_URL", "")
DEFAULT_DEVICE_ID = "ESP32-CAM-NODE-01"
