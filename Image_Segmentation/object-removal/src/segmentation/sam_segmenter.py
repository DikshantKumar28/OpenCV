"""
src/segmentation/sam_segmenter.py
----------------------------------
Optional SAM (Segment Anything Model) refiner.

Uses the Ultralytics SAM interface so weights auto-download on first run.
If SAM is disabled in config or fails to load, all methods return the
original YOLO mask unchanged — no errors are raised.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import USE_SAM, SAM_MODEL_NAME
from src.utils.mask_utils import normalize_yolo_mask, bbox_to_mask

logger = logging.getLogger(__name__)


class SamSegmenter:
    """
    Wraps the Ultralytics SAM model for optional mask refinement.

    Usage:
        sam = SamSegmenter()
        refined_mask = sam.refine_mask(image, bbox, yolo_mask)
    """

    def __init__(self):
        self.model   = None
        self.enabled = USE_SAM
        if self.enabled:
            self._load_model()

    # ──────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Load SAM via Ultralytics. Weights are auto-downloaded on first call.
        If loading fails for any reason, SAM is silently disabled.
        """
        try:
            from ultralytics import SAM
            logger.info(f"Loading SAM model: {SAM_MODEL_NAME}")
            self.model   = SAM(SAM_MODEL_NAME)
            self.enabled = True
            logger.info("SAM model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"SAM could not be loaded ({e}). "
                "Object removal will use YOLO masks only."
            )
            self.model   = None
            self.enabled = False

    # ──────────────────────────────────────────────────────────────────
    # Mask refinement
    # ──────────────────────────────────────────────────────────────────
    def refine_mask(
        self,
        image: np.ndarray,
        bbox: list[int],
        fallback_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Use SAM to produce a refined segmentation mask for the given bbox.

        Parameters
        ----------
        image         : BGR image (H, W, 3)
        bbox          : [x1, y1, x2, y2] in pixel coordinates
        fallback_mask : uint8 (H, W) mask to return if SAM is unavailable

        Returns
        -------
        uint8 binary mask (H, W) at the original image resolution.
        """
        orig_h, orig_w = image.shape[:2]

        if not self.enabled or self.model is None:
            # SAM not available — return fallback or bbox mask
            if fallback_mask is not None:
                return fallback_mask
            return bbox_to_mask(bbox, orig_h, orig_w)

        try:
            # SAM expects bboxes as [[x1, y1, x2, y2]]
            results = self.model.predict(
                source=image,
                bboxes=[bbox],
                verbose=False,
            )

            if results and results[0].masks is not None:
                raw = results[0].masks.data[0].cpu().numpy()  # (H', W') float
                refined = normalize_yolo_mask(raw, orig_h, orig_w)
                logger.debug("SAM mask refinement successful.")
                return refined

        except Exception as e:
            logger.warning(f"SAM inference failed: {e}. Using fallback mask.")

        # If SAM inference fails, use the YOLO mask
        if fallback_mask is not None:
            return fallback_mask
        return bbox_to_mask(bbox, orig_h, orig_w)

    def is_available(self) -> bool:
        """Return True if SAM is loaded and ready."""
        return self.enabled and self.model is not None
