"""
src/utils/visualization.py
--------------------------
Draw bounding boxes, class labels, object IDs, and mask overlays
on images for detection preview sent back to the frontend.
"""

import cv2
import numpy as np
import random

# A fixed set of bright colours for up to 80 object classes
_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given class ID."""
    rgb = _PALETTE[class_id % len(_PALETTE)]
    return (rgb[2], rgb[1], rgb[0])   # convert to BGR


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    draw_masks: bool = True,
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """
    Draw bounding boxes, object IDs, class names, confidence scores,
    and optional filled mask overlays on a copy of the image.

    detections: list of detection dicts with keys:
        object_id, class_id, class_name, confidence, bbox, mask (optional)
    Returns: annotated BGR image.
    """
    annotated = image.copy()
    overlay   = image.copy()

    for det in detections:
        oid        = det["object_id"]
        class_id   = det.get("class_id", 0)
        class_name = det.get("class_name", "object")
        conf       = det.get("confidence", 0.0)
        bbox       = det.get("bbox", [0, 0, 0, 0])    # [x1, y1, x2, y2]
        mask       = det.get("mask", None)             # uint8 (H, W) or None

        color = get_color(class_id)
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # ── Filled mask overlay ──────────────────────────────────────────
        if draw_masks and mask is not None and mask.size > 0:
            color_mask = np.zeros_like(image, dtype=np.uint8)
            color_mask[mask > 127] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, mask_alpha, 0)

        # ── Bounding box ─────────────────────────────────────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ── Label background ─────────────────────────────────────────────
        label = f"#{oid} {class_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 4, y1), color, cv2.FILLED)

        # ── Label text ───────────────────────────────────────────────────
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend mask overlay into annotated image
    if draw_masks:
        annotated = cv2.addWeighted(overlay, mask_alpha, annotated, 1 - mask_alpha, 0)

    return annotated


def draw_mask_only(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 80),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a single binary mask (white = selected area) on the image.
    Useful for showing the user what will be removed.
    """
    annotated = image.copy()
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[mask > 127] = color
    return cv2.addWeighted(annotated, 1.0, color_layer, alpha, 0)


def side_by_side(original: np.ndarray, result: np.ndarray) -> np.ndarray:
    """
    Concatenate original and result horizontally with a divider line.
    Both images are resized to the same height if they differ.
    """
    h1, w1 = original.shape[:2]
    h2, w2 = result.shape[:2]

    if h1 != h2:
        result = cv2.resize(result, (w2, h1))

    divider = np.full((h1, 4, 3), 200, dtype=np.uint8)  # grey divider
    return np.hstack([original, divider, result])
