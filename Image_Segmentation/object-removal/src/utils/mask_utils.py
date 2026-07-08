"""
src/utils/mask_utils.py
-----------------------
All mask manipulation helpers: combining, resizing, dilating,
smoothing, and converting masks to preview images.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MASK_DILATION_KERNEL, MASK_BLUR_KERNEL


def create_empty_mask(height: int, width: int) -> np.ndarray:
    """Return a black (all-zero) uint8 mask of shape (H, W)."""
    return np.zeros((height, width), dtype=np.uint8)


def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    """
    Combine a list of binary masks (0/255) into a single union mask.
    All masks must be the same shape.
    """
    if not masks:
        raise ValueError("No masks provided to combine.")
    combined = create_empty_mask(*masks[0].shape[:2])
    for m in masks:
        if m.shape[:2] != combined.shape[:2]:
            m = resize_mask(m, combined.shape[0], combined.shape[1])
        combined = cv2.bitwise_or(combined, binarize_mask(m))
    return combined


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Resize a binary mask to (height, width) using nearest-neighbour
    interpolation so pixel values stay 0 or 255.
    """
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def dilate_mask(mask: np.ndarray, kernel_size: int = MASK_DILATION_KERNEL, iterations: int = 1) -> np.ndarray:
    """
    Expand the mask outward by kernel_size pixels.
    Useful for inpainting to ensure the full object edge is covered.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Shrink the mask inward — useful to remove thin noise edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def smooth_mask(mask: np.ndarray, blur_size: int = MASK_BLUR_KERNEL) -> np.ndarray:
    """
    Apply Gaussian blur to feather mask edges.
    blur_size must be odd. If even, it's incremented by 1.
    """
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return blurred


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert a grayscale / soft mask back to a hard binary mask (0 or 255).
    """
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a binary mask so the object region remains solid.
    """
    binary = binarize_mask(mask)
    h, w = binary.shape[:2]
    flood_filled = binary.copy()
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask_flood, (0, 0), 255)
    filled = cv2.bitwise_not(flood_filled)
    return cv2.bitwise_or(binary, filled)


def get_mask_area_ratio(mask: np.ndarray, image_shape: tuple) -> float:
    """
    Calculate the area ratio of the mask relative to the full image.
    """
    if mask is None or mask.size == 0:
        return 0.0
    area = float(np.count_nonzero(binarize_mask(mask) > 0))
    full_area = float(image_shape[0] * image_shape[1])
    return area / full_area if full_area > 0 else 0.0


def extract_boundary(mask: np.ndarray, distance: int = 8) -> np.ndarray:
    """
    Create a boundary mask by subtracting an eroded mask from a dilated mask.
    """
    binary = binarize_mask(mask)
    dilated = dilate_mask(binary, kernel_size=distance, iterations=1)
    eroded = erode_mask(binary, kernel_size=max(3, distance // 2), iterations=1)
    boundary = cv2.subtract(dilated, eroded)
    return binarize_mask(boundary)


def bbox_to_mask(bbox: list[int], height: int, width: int) -> np.ndarray:
    """
    Create a rectangular binary mask from a bounding box.
    bbox format: [x1, y1, x2, y2] in pixel coordinates.
    Used as fallback when YOLO returns no polygon/segment mask.
    """
    mask = create_empty_mask(height, width)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    mask[y1:y2, x1:x2] = 255
    return mask


def polygon_to_mask(polygon: list[list[float]], height: int, width: int) -> np.ndarray:
    """
    Convert a list of [x, y] polygon points to a filled binary mask.
    Points are assumed to be in pixel coordinates.
    """
    mask = create_empty_mask(height, width)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_to_preview(mask: np.ndarray, color: tuple = (0, 255, 100), alpha: float = 0.5) -> np.ndarray:
    """
    Convert a binary mask into an RGBA overlay for frontend preview.
    Returns an RGBA uint8 image of shape (H, W, 4).
    color: RGB tuple for the mask colour.
    """
    h, w = mask.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask > 127, :3] = color
    overlay[mask > 127, 3] = int(alpha * 255)
    return overlay


def mask_to_base64_preview(mask: np.ndarray) -> str:
    """
    Encode a binary mask as a base64 PNG string for sending to the frontend.
    """
    import base64
    rgba = mask_to_preview(mask)
    success, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def normalize_yolo_mask(raw_mask: np.ndarray, orig_height: int, orig_width: int) -> np.ndarray:
    """
    YOLO segmentation masks may be returned at a lower resolution (e.g. 160×160).
    This function resizes them to the original image dimensions and binarizes.
    raw_mask: float32 array with values 0.0–1.0 from YOLO inference.
    """
    mask_uint8 = (raw_mask * 255).astype(np.uint8)
    resized = resize_mask(mask_uint8, orig_height, orig_width)
    return binarize_mask(resized)


def expand_and_refine_mask(
    mask: np.ndarray,
    image_shape: tuple,
    kernel_size: int = 9,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refines and expands a mask to cover object boundaries:
    1. Resize input mask to original image shape.
    2. Binarize (0/255).
    3. Fill holes and remove tiny noise.
    4. Dilate using an elliptical structuring element of size kernel_size for iterations.
    5. Apply morphological close (cv2.MORPH_CLOSE).
    6. Return both the binary closed mask and a Gaussian-blurred mask.
    """
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = resize_mask(mask, h, w)
    binary = binarize_mask(mask)

    filled = fill_mask_holes(binary)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, open_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(opened, dilate_kernel, iterations=iterations)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, dilate_kernel)

    blur_size = 2 * kernel_size + 3
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = smooth_mask(closed, blur_size)
    return closed, blurred


# ─────────────────────────────────────────────
# AI Inpainting specific mask & crop utilities
# ─────────────────────────────────────────────

def expand_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand the mask evenly by `pixels`."""
    if pixels <= 0:
        return mask
    return dilate_mask(mask, kernel_size=pixels, iterations=1)

def smooth_mask_edges(mask: np.ndarray, feather_pixels: int) -> np.ndarray:
    """Smooth the edges of the mask for blending."""
    return smooth_mask(mask, blur_size=feather_pixels * 2 + 1)

def prepare_ai_inpaint_mask(mask: np.ndarray, image_shape: tuple, expand_pixels: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (expanded_mask, blurred_mask) for AI inpainting.
    Uses fill_holes and expands.
    """
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = resize_mask(mask, h, w)

    binary = binarize_mask(mask)
    filled = fill_mask_holes(binary)
    expanded = expand_mask(filled, expand_pixels)
    # The blurred mask is used for blending later
    blurred = smooth_mask_edges(expanded, max(12, expand_pixels // 2))
    return expanded, blurred

def get_mask_bbox(mask: np.ndarray, padding: int, image_shape: tuple) -> list[int]:
    """
    Get [x1, y1, x2, y2] bounding box of the mask, expanded by padding.
    """
    h, w = image_shape[:2]
    binary = binarize_mask(mask)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return [0, 0, w, h]  # Fallback to full image
    
    x, y, bw, bh = cv2.boundingRect(coords)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    return [x1, y1, x2, y2]

def crop_to_mask_bbox(image: np.ndarray, mask: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Crops both image and mask to the padded bounding box of the mask.
    Returns (cropped_image, cropped_mask, bbox).
    """
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = resize_mask(mask, h, w)

    bbox = get_mask_bbox(mask, padding, image.shape)
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], bbox

def paste_crop_back(original: np.ndarray, inpainted_crop: np.ndarray, bbox: list[int], blend_mask: np.ndarray = None) -> np.ndarray:
    """
    Pastes the inpainted crop back into the original image at the bbox location.
    If blend_mask is provided (in crop coordinates), uses it to blend.
    """
    x1, y1, x2, y2 = bbox
    result = original.copy()
    crop_h = max(0, y2 - y1)
    crop_w = max(0, x2 - x1)
    if inpainted_crop.shape[:2] != (crop_h, crop_w):
        inpainted_crop = cv2.resize(inpainted_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    
    if blend_mask is not None:
        if blend_mask.shape[:2] != (crop_h, crop_w):
            blend_mask = resize_mask(blend_mask, crop_h, crop_w)
        crop_mask_float = blend_mask.astype(np.float32) / 255.0
        if len(crop_mask_float.shape) == 2:
            crop_mask_float = np.expand_dims(crop_mask_float, axis=-1)
        
        target_roi = result[y1:y2, x1:x2].astype(np.float32)
        source_crop = inpainted_crop.astype(np.float32)
        
        blended_roi = (source_crop * crop_mask_float) + (target_roi * (1.0 - crop_mask_float))
        result[y1:y2, x1:x2] = blended_roi.astype(np.uint8)
    else:
        result[y1:y2, x1:x2] = inpainted_crop
        
    return result

