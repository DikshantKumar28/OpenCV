"""
src/alignment/image_aligner.py
--------------------------------
Aligns a reference image to a target image using ORB feature
matching + RANSAC homography (OpenCV).

If alignment fails (too few matches), None is returned so the caller
can safely skip that reference image.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ORB_MAX_FEATURES, HOMOGRAPHY_MIN_MATCH

logger = logging.getLogger(__name__)


class ImageAligner:
    """
    Aligns one or more reference images to a target image.

    Usage:
        aligner = ImageAligner()
        aligned = aligner.align(reference_bgr, target_bgr)
        # aligned is None if alignment failed
    """

    def __init__(
        self,
        max_features: int = ORB_MAX_FEATURES,
        min_match_count: int = HOMOGRAPHY_MIN_MATCH,
    ):
        self.max_features     = max_features
        self.min_match_count  = min_match_count

        # ORB detector + brute-force Hamming matcher
        self.orb      = cv2.ORB_create(nfeatures=max_features)
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def align(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray | None:
        """
        Warp `reference` so that its features align with `target`.

        Parameters
        ----------
        reference : BGR image to warp
        target    : BGR image to align against (stays fixed)

        Returns
        -------
        Warped BGR image of the same size as target, or None on failure.
        """
        h, w = target.shape[:2]

        # Convert both to greyscale for feature detection
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target,    cv2.COLOR_BGR2GRAY)

        # ── Detect & describe keypoints ───────────────────────────────
        kp_ref, des_ref = self.orb.detectAndCompute(ref_gray, None)
        kp_tgt, des_tgt = self.orb.detectAndCompute(tgt_gray, None)

        if des_ref is None or des_tgt is None:
            logger.warning("ORB: no descriptors found — skipping alignment.")
            return None

        # ── Match descriptors with KNN ratio test ─────────────────────
        try:
            matches = self.matcher.knnMatch(des_ref, des_tgt, k=2)
        except cv2.error as e:
            logger.warning(f"BFMatcher failed: {e} — skipping alignment.")
            return None

        # Lowe's ratio test to keep only good matches
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_match_count:
            logger.warning(
                f"ORB: only {len(good_matches)} good matches "
                f"(need {self.min_match_count}) — skipping alignment."
            )
            return None

        # ── Compute homography ────────────────────────────────────────
        src_pts = np.float32(
            [kp_ref[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_tgt[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            logger.warning("Homography estimation failed — skipping alignment.")
            return None

        inliers = int(mask.ravel().sum()) if mask is not None else 0
        logger.debug(f"Homography found with {inliers} inliers.")

        # ── Warp reference to target space ────────────────────────────
        aligned = cv2.warpPerspective(reference, H, (w, h))
        return aligned

    def align_multiple(
        self,
        references: list[np.ndarray],
        target: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Align a list of reference images to the target.
        Failed alignments are silently skipped (not included in result).

        Returns a list of successfully aligned BGR images.
        """
        aligned_list = []
        for i, ref in enumerate(references):
            result = self.align(ref, target)
            if result is not None:
                aligned_list.append(result)
                logger.info(f"Reference image {i+1}: aligned successfully.")
            else:
                logger.warning(f"Reference image {i+1}: alignment failed — skipped.")
        return aligned_list
