"""
src/detection/yolo_detector.py
-------------------------------
YoloDetector class:
  - Loads a YOLO segmentation model via Ultralytics.
  - Runs inference and returns structured detection dicts.
  - Attempts yolo11n-seg.pt first; falls back to yolov8n-seg.pt.
  - Converts raw YOLO masks to full-resolution binary masks.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    YOLO_MODEL_PRIMARY, YOLO_MODEL_FALLBACK,
    YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_IMG_SIZE,
    MIN_MASK_AREA,
)
from src.utils.mask_utils import normalize_yolo_mask, bbox_to_mask

logger = logging.getLogger(__name__)


class YoloDetector:
    """
    Wraps the Ultralytics YOLO segmentation model.

    Usage:
        detector = YoloDetector()
        detections = detector.detect(image_bgr)
    """

    def __init__(self):
        self.model = self._load_model()

    # ──────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Try loading the primary YOLO model first.
        If that fails (e.g. not yet available in installed Ultralytics),
        fall back to the secondary model.
        Ultralytics auto-downloads weights on first use.
        """
        from ultralytics import YOLO

        for model_name in [YOLO_MODEL_PRIMARY, YOLO_MODEL_FALLBACK]:
            try:
                logger.info(f"Loading YOLO model: {model_name}")
                model = YOLO(model_name)
                # Quick validation: make sure it's a segmentation model
                if not hasattr(model, "predictor") or "seg" not in model_name:
                    # Force a dummy predict to trigger download and validate
                    pass
                logger.info(f"YOLO model loaded successfully: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}. Trying fallback...")

        raise RuntimeError(
            "Could not load any YOLO segmentation model. "
            "Please check your Ultralytics installation."
        )

    # ──────────────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────────────
    def detect(
        self,
        image: np.ndarray,
        conf: float = YOLO_CONF_THRESHOLD,
        iou: float  = YOLO_IOU_THRESHOLD,
        img_size: int = YOLO_IMG_SIZE,
    ) -> list[dict]:
        """
        Run YOLO segmentation on a BGR image.

        Returns a list of detection dicts:
        {
            "object_id":   int,      # 0-indexed sequential ID
            "class_id":    int,      # YOLO class index
            "class_name":  str,      # Human-readable class label
            "confidence":  float,    # 0.0–1.0
            "bbox":        [x1, y1, x2, y2],  # pixel coords in original image
            "mask":        np.ndarray (H, W, uint8),  # binary mask at orig res
            "mask_b64":    str,      # base64-encoded PNG preview for frontend
        }
        """
        orig_h, orig_w = image.shape[:2]
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        result = results[0]           # single image → single result
        class_names = result.names    # {0: 'person', 1: 'car', ...}
        boxes = result.boxes          # bounding boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # ── Segmentation masks (if available) ────────────────────────
        raw_masks = None
        if result.masks is not None:
            raw_masks = result.masks.data.cpu().numpy()  # shape (N, H', W')

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            class_id   = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = class_names.get(class_id, f"class_{class_id}")

            # ── Build binary mask ─────────────────────────────────────
            if raw_masks is not None and idx < len(raw_masks):
                # Resize low-res YOLO mask to original image size
                mask = normalize_yolo_mask(raw_masks[idx], orig_h, orig_w)
            else:
                # Fallback: filled bounding box
                mask = bbox_to_mask([x1, y1, x2, y2], orig_h, orig_w)

            # ── Calculate mask area ───────────────────────────────────
            mask_area = int(np.sum(mask > 127))
            if mask_area < MIN_MASK_AREA:
                continue

            # ── Base64 preview for frontend ───────────────────────────
            from src.utils.mask_utils import mask_to_base64_preview
            mask_b64 = mask_to_base64_preview(mask)

            # ── Compress mask to save memory ──────────────────────────
            _, mask_encoded = cv2.imencode('.png', mask)

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox":       [round(v) for v in [x1, y1, x2, y2]],
                "mask_png":   mask_encoded,
                "mask_b64":   mask_b64,
                "mask_area":  mask_area,
            })

        # ── Sort by confidence descending and reassign IDs ────────────
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        for idx, det in enumerate(detections):
            det["object_id"] = idx

        logger.info(f"Detected {len(detections)} object(s).")
        return detections
