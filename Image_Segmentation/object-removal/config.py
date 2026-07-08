"""
config.py
---------
Central configuration for the object-removal application.
All paths, model names, and tunable thresholds live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base directories
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR        = BASE_DIR / "data"
INPUT_SINGLE    = DATA_DIR / "input" / "single"
INPUT_MULTIPLE  = DATA_DIR / "input" / "multiple"
MASKS_DIR       = DATA_DIR / "masks"
OUTPUT_DIR      = DATA_DIR / "output"

MODELS_DIR      = BASE_DIR / "models"
YOLO_MODEL_DIR  = MODELS_DIR / "yolo"
SAM_MODEL_DIR   = MODELS_DIR / "sam"

# ─────────────────────────────────────────────
# Auto-create all required directories on import
# ─────────────────────────────────────────────
for _dir in [INPUT_SINGLE, INPUT_MULTIPLE, MASKS_DIR, OUTPUT_DIR,
             YOLO_MODEL_DIR, SAM_MODEL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Model configuration
# ─────────────────────────────────────────────

# YOLO segmentation model — Ultralytics will auto-download on first run.
# Primary choice: yolo11n-seg.pt  (latest lightweight seg model)
# Fallback:       yolov8n-seg.pt  (stable, widely available)
YOLO_MODEL_PRIMARY  = "yolo11n-seg.pt"
YOLO_MODEL_FALLBACK = "yolov8n-seg.pt"

# SAM model — loaded via Ultralytics SAM interface.
# Set USE_SAM = False to skip SAM and rely solely on YOLO masks.
USE_SAM         = True
SAM_MODEL_NAME  = "sam2.1_b.pt"     # Ultralytics auto-downloads this

# ─────────────────────────────────────────────
# Detection thresholds
# ─────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.15   # Minimum confidence score to keep a detection
YOLO_IOU_THRESHOLD  = 0.45   # Non-maximum suppression IoU threshold
YOLO_IMG_SIZE       = 640    # Inference image size (pixels)

# ─────────────────────────────────────────────
# Mask processing
# ─────────────────────────────────────────────
MASK_DILATION_KERNEL   = 15    # Legacy kernel size for dilating mask
MASK_BLUR_KERNEL       = 21    # Gaussian blur kernel for mask feathering
MASK_DILATE_ITERATIONS = 4
MASK_DILATE_KERNEL     = 9
INPAINT_RADIUS         = 15
MIN_MASK_AREA          = 100
SMALL_MASK_AREA_RATIO  = 0.025
MEDIUM_MASK_AREA_RATIO = 0.08
LARGE_MASK_AREA_RATIO  = 0.12

# ─────────────────────────────────────────────
# AI Inpainting
# ─────────────────────────────────────────────
try:
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _device = "cpu"

AI_INPAINTING_AVAILABLE = True  # We will dynamically check later, but set True globally if intended
AI_INPAINTING_ENABLED = True
AI_INPAINTING_BACKEND = "lama"
AI_INPAINTING_DEVICE = _device
AI_INPAINTING_MIN_MASK_RATIO = 0.025
AI_INPAINTING_STRENGTH = 0.95
AI_INPAINTING_PADDING = 64
MASK_EXPAND_PIXELS_AI = 24
MASK_FEATHER_PIXELS = 12
COLOR_MATCH_ENABLED = True

# ─────────────────────────────────────────────
# Multi-image removal pipeline configuration
# ─────────────────────────────────────────────
MULTI_SMALL_MASK_RATIO        = 0.025   # Small object threshold for multi-image
MULTI_MEDIUM_MASK_RATIO       = 0.12    # Medium object threshold
MULTI_LARGE_MASK_RATIO        = 0.20    # Large object threshold (AI required beyond this)
MULTI_MAX_FEATHER_RADIUS      = 15      # Max boundary feathering radius
MULTI_SEAM_INPAINT_RADIUS     = 3       # Seam cleanup inpaint radius (boundary only)
REFERENCE_MIN_VALID_COUNT     = 1       # Minimum aligned references required
REFERENCE_QUALITY_THRESHOLD   = 0.60    # Min quality score to use reference replacement

# ─────────────────────────────────────────────
# Multi-image alignment
# ─────────────────────────────────────────────
ORB_MAX_FEATURES     = 5000   # Max ORB keypoints to detect
HOMOGRAPHY_MIN_MATCH = 10     # Minimum good matches required for homography

# ─────────────────────────────────────────────
# Upload / server settings
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_SIZE_MB  = 50     # Max single-file upload size in MB
API_HOST            = "0.0.0.0"
API_PORT            = 8000
CORS_ORIGINS        = ["http://localhost:5173", "http://127.0.0.1:5173"]
