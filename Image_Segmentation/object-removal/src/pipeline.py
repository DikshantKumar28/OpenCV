"""
src/pipeline.py
----------------
High-level orchestration layer.

All FastAPI endpoint handlers call these three functions:
  - detect_objects()
  - remove_from_single_image()
  - remove_from_multiple_images()

Each function handles loading, calling sub-modules, saving outputs,
and returning clean result dicts ready for JSON serialisation.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, MASKS_DIR, USE_SAM
)
from src.detection.yolo_detector import YoloDetector
from src.segmentation.sam_segmenter import SamSegmenter
from src.removal.single_image_remover import SingleImageRemover
from src.removal.multi_image_remover import MultiImageRemover
from src.utils.image_io import save_image_cv2, load_image_cv2
from src.utils.mask_utils import combine_masks
from src.utils.visualization import draw_detections

logger = logging.getLogger(__name__)

# ── Module-level singleton models (loaded once, reused across requests) ──
_detector = None
_sam      = None

def _get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector()
    return _detector

def _get_sam() -> SamSegmenter:
    global _sam
    if _sam is None:
        _sam = SamSegmenter()
    return _sam


# ═══════════════════════════════════════════════════════════════════════
# 1. Object detection
# ═══════════════════════════════════════════════════════════════════════
def detect_objects(image_path: str | Path, conf: float = None) -> dict:
    """
    Run YOLO (+ optional SAM) detection on the given image.

    Returns
    -------
    {
        "detections": [
            {
                "object_id":  int,
                "class_name": str,
                "confidence": float,
                "bbox":       [x1, y1, x2, y2],
                "mask_b64":   str (base64 PNG)
            },
            ...
        ],
        "preview_path": str  # path to annotated preview image saved to disk
    }
    """
    image = load_image_cv2(image_path)
    detector = _get_detector()
    sam      = _get_sam() if USE_SAM else None

    if conf is not None:
        raw_detections = detector.detect(image, conf=conf)
    else:
        raw_detections = detector.detect(image)

    # Optional SAM refinement for each detection
    if sam and sam.is_available():
        for det in raw_detections:
            mask = cv2.imdecode(det["mask_png"], cv2.IMREAD_GRAYSCALE) if "mask_png" in det else det.get("mask")
            refined = sam.refine_mask(image, det["bbox"], mask)
            
            # Re-encode compressed mask
            _, encoded = cv2.imencode('.png', refined)
            det["mask_png"] = encoded
            det.pop("mask", None) # Remove uncompressed mask if present
            
            # Re-encode base64 preview with refined mask
            from src.utils.mask_utils import mask_to_base64_preview
            det["mask_b64"] = mask_to_base64_preview(refined)

    # Build preview image and save to output folder
    annotated = draw_detections(image, raw_detections, draw_masks=True)
    preview_name = f"preview_{uuid.uuid4().hex}.jpg"
    preview_path = OUTPUT_DIR / preview_name
    save_image_cv2(annotated, preview_path)

    # Strip numpy arrays before returning (not JSON-serialisable)
    serialisable = []
    for det in raw_detections:
        serialisable.append({
            "object_id":  det["object_id"],
            "class_id":   det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox":       det["bbox"],
            "mask_b64":   det["mask_b64"],
        })

    return {
        "detections":   serialisable,
        "raw_detections": raw_detections,
        "preview_path": str(preview_path),
        "preview_name": preview_name,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Single-image removal
# ═══════════════════════════════════════════════════════════════════════
def remove_from_single_image(
    image_path: str | Path,
    detections: list[dict],
    selected_ids: list[int],
    dilate_kernel: int = None,
    dilate_iterations: int = None,
    inpaint_radius: int = None,
    removal_mode: str = "auto",
    use_strong_inpaint: bool = True,
) -> dict:
    """
    Remove selected objects from a single image via inpainting.

    Parameters
    ----------
    image_path   : path to original image
    detections   : full detection list returned by detect_objects()
                   (must still contain 'mask' numpy arrays — stored server-side)
    selected_ids : list of object_ids chosen by the user

    Returns
    -------
    { "output_path": str, "output_name": str, "mask_path": str, "report": dict }
    """
    image = load_image_cv2(image_path)
    masks = _collect_masks(image, detections, selected_ids)

    remover = SingleImageRemover()
    
    kwargs = {
        "removal_mode": removal_mode,
        "use_strong_inpaint": use_strong_inpaint,
    }
    if dilate_kernel is not None: kwargs['dilate_kernel'] = dilate_kernel
    if dilate_iterations is not None: kwargs['dilate_iterations'] = dilate_iterations
    if inpaint_radius is not None: kwargs['inpaint_radius'] = inpaint_radius
    
    result, combined_mask, report = remover.remove(image, masks, **kwargs)

    # Save output
    out_name = f"result_{uuid.uuid4().hex}.jpg"
    out_path = save_image_cv2(result, OUTPUT_DIR / out_name)

    # Save combined mask
    mask_name = f"mask_{uuid.uuid4().hex}.png"
    mask_path = save_image_cv2(combined_mask, MASKS_DIR / mask_name)

    logger.info(f"Single-image result saved: {out_path}")
    return {
        "output_path": str(out_path),
        "output_name": out_name,
        "mask_path":   str(mask_path),
        "report": report,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Multi-image removal
# ═══════════════════════════════════════════════════════════════════════
def remove_from_multiple_images(
    target_path: str | Path,
    reference_paths: list[str | Path],
    detections: list[dict],
    selected_ids: list[int],
    dilate_kernel: int = None,
    dilate_iterations: int = None,
    inpaint_radius: int = None,
    removal_mode: str = "auto",
    use_strong_inpaint: bool = True,
) -> dict:
    """
    Remove selected objects using reference images for background reconstruction.

    Parameters
    ----------
    target_path      : path to the target (first uploaded) image
    reference_paths  : paths to reference / duplicate images
    detections       : detection list (with masks) for the target image
    selected_ids     : list of object_ids to remove

    Returns
    -------
    { "output_path": str, "output_name": str, "mask_path": str, "report": dict }
    """
    target     = load_image_cv2(target_path)
    references = [load_image_cv2(p) for p in reference_paths]
    masks      = _collect_masks(target, detections, selected_ids)

    remover = MultiImageRemover()
    
    kwargs = {
        "removal_mode": removal_mode,
        "use_strong_inpaint": use_strong_inpaint,
    }
    if dilate_kernel is not None: kwargs['dilate_kernel'] = dilate_kernel
    if dilate_iterations is not None: kwargs['dilate_iterations'] = dilate_iterations
    if inpaint_radius is not None: kwargs['inpaint_radius'] = inpaint_radius
    
    result, combined_mask, report = remover.remove(target, references, masks, **kwargs)

    # Save output
    out_name = f"result_{uuid.uuid4().hex}.jpg"
    out_path = save_image_cv2(result, OUTPUT_DIR / out_name)

    # Save combined mask
    mask_name = f"mask_{uuid.uuid4().hex}.png"
    mask_path = save_image_cv2(combined_mask, MASKS_DIR / mask_name)

    logger.info(f"Multi-image result saved: {out_path}")
    return {
        "output_path": str(out_path),
        "output_name": out_name,
        "mask_path":   str(mask_path),
        "report": report,
    }


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════
def _collect_masks(
    image: np.ndarray,
    detections: list[dict],
    selected_ids: list[int],
) -> list[np.ndarray]:
    """
    Pull numpy masks out of the detection list for the given object IDs.
    If a detection only has mask_b64 (numpy mask was not kept), reconstruct
    from the bounding box as fallback.
    """
    from src.utils.mask_utils import bbox_to_mask
    h, w = image.shape[:2]
    masks = []

    det_by_id = {d["object_id"]: d for d in detections}
    for oid in selected_ids:
        det = det_by_id.get(oid)
        if det is None:
            logger.warning(f"object_id {oid} not found in detections — skipped.")
            continue

        if "mask_png" in det and det["mask_png"] is not None:
            mask = cv2.imdecode(det["mask_png"], cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        elif "mask" in det and isinstance(det["mask"], np.ndarray):
            masks.append(det["mask"])
        else:
            # Fallback: reconstruct from bbox
            masks.append(bbox_to_mask(det["bbox"], h, w))

    return masks
