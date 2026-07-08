"""
src/blending/blender.py
------------------------
Smooth blending of a replacement patch into the target image.

Strategy:
1. Try cv2.seamlessClone (Poisson blending) — best quality.
2. Fall back to feathered alpha blending if seamlessClone fails
   (e.g. mask too close to image edge).
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MASK_BLUR_KERNEL

logger = logging.getLogger(__name__)


class Blender:
    """
    Blends a source patch into a target image using the provided mask.

    Usage:
        blender = Blender()
        result = blender.blend(target, source, mask)
    """

    def _normalize_blend_inputs(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensure source and mask match the target crop before OpenCV blending.
        """
        h, w = target.shape[:2]

        if source.shape[:2] != (h, w):
            source = cv2.resize(source, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        _, mask = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        return target, source, mask

    def blend(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        feather_radius: int = MASK_BLUR_KERNEL,
    ) -> np.ndarray:
        """
        Blend `source` pixels into `target` where `mask` is non-zero.

        Parameters
        ----------
        target         : BGR destination image
        source         : BGR source image (same size as target)
        mask           : uint8 binary mask (H, W) — white = region to blend
        feather_radius : Gaussian blur size for alpha feathering

        Returns
        -------
        Blended BGR image.
        """
        target, source, mask = self._normalize_blend_inputs(target, source, mask)

        # First try the high-quality Poisson seamless clone
        result = self._seamless_clone(target, source, mask)
        if result is not None:
            return result

        # Fall back to alpha blend with feathered edges
        logger.debug("Using alpha-blend fallback.")
        return self._alpha_blend(target, source, mask, feather_radius)

    # ──────────────────────────────────────────────────────────────────
    # Strategy 1: Poisson / seamless clone
    # ──────────────────────────────────────────────────────────────────
    def _seamless_clone(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        """
        Attempt cv2.seamlessClone.
        Returns None if the mask centroid is too close to the image border
        (which causes OpenCV to throw an error).
        """
        target, source, mask = self._normalize_blend_inputs(target, source, mask)
        h, w = target.shape[:2]

        # Compute mask centroid for seamlessClone center point
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # seamlessClone fails if the bounding box of the mask touches the border
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None

        margin = 5
        if (xs.min() < margin or xs.max() > w - margin or
                ys.min() < margin or ys.max() > h - margin):
            logger.debug("Mask too close to border — skipping seamlessClone.")
            return None

        try:
            # seamlessClone needs uint8 mask with value exactly 255
            clone_mask = (mask > 127).astype(np.uint8) * 255
            result = cv2.seamlessClone(
                source, target, clone_mask, (cx, cy), cv2.NORMAL_CLONE
            )
            return result
        except cv2.error as e:
            logger.debug(f"seamlessClone error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    # Strategy 2: Feathered alpha blend
    # ──────────────────────────────────────────────────────────────────
    def _alpha_blend(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        feather_radius: int,
    ) -> np.ndarray:
        """
        Blend source into target using a Gaussian-feathered mask as alpha.
        """
        target, source, mask = self._normalize_blend_inputs(target, source, mask)

        if feather_radius % 2 == 0:
            feather_radius += 1

        # Feather the mask edges with Gaussian blur
        alpha = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (feather_radius, feather_radius),
            0
        )
        alpha_3ch = alpha[:, :, np.newaxis]           # shape (H, W, 1)

        target_f = target.astype(np.float32)
        source_f = source.astype(np.float32)

        blended = source_f * alpha_3ch + target_f * (1.0 - alpha_3ch)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def feathered_blend(
        self,
        base: np.ndarray,
        replacement: np.ndarray,
        mask: np.ndarray,
        blur_radius: int = 21,
    ) -> np.ndarray:
        """
        Blend replacement into base using a Gaussian-feathered mask as alpha.
        """
        base, replacement, mask = self._normalize_blend_inputs(base, replacement, mask)

        if blur_radius % 2 == 0:
            blur_radius += 1

        # Feather the mask edges with Gaussian blur
        alpha = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (blur_radius, blur_radius),
            0
        )
        # Ensure alpha has shape (H, W, 1)
        if len(alpha.shape) == 2:
            alpha_3ch = alpha[:, :, np.newaxis]
        else:
            alpha_3ch = alpha

        base_f = base.astype(np.float32)
        replacement_f = replacement.astype(np.float32)

        blended = base_f * (1.0 - alpha_3ch) + replacement_f * alpha_3ch
        return np.clip(blended, 0, 255).astype(np.uint8)

    def feather_boundary_blend(self, original: np.ndarray, replacement: np.ndarray, mask: np.ndarray, feather_pixels: int) -> np.ndarray:
        return self.feathered_blend(original, replacement, mask, blur_radius=feather_pixels * 2 + 1)

    def match_color_to_surroundings(self, original: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Adjusts the color/brightness of the `replacement` to match `original` outside the `mask`.
        This is a simplified color matching using mean/std adjustment in LAB color space.
        """
        original, replacement, mask = self._normalize_blend_inputs(original, replacement, mask)

        # Ensure mask is binary
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate mask slightly to get the immediate surroundings (the 'band')
        dilated = cv2.dilate(bin_mask, np.ones((15, 15), np.uint8), iterations=1)
        band = cv2.bitwise_xor(dilated, bin_mask)
        
        # If the band is empty, skip color match
        if np.count_nonzero(band) == 0:
            return replacement

        # Convert to LAB for luminance matching
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        repl_lab = cv2.cvtColor(replacement, cv2.COLOR_BGR2LAB)

        orig_mean, orig_std = cv2.meanStdDev(orig_lab, mask=band)
        repl_mean, repl_std = cv2.meanStdDev(repl_lab, mask=bin_mask)
        
        # Adjust only L channel or all channels. Let's do all channels for color shift.
        repl_f = repl_lab.astype(np.float32)
        
        for i in range(3):
            if repl_std[i][0] == 0:
                continue
            repl_f[:, :, i] = ((repl_f[:, :, i] - repl_mean[i][0]) * (orig_std[i][0] / repl_std[i][0])) + orig_mean[i][0]
            
        repl_f = np.clip(repl_f, 0, 255).astype(np.uint8)
        matched_bgr = cv2.cvtColor(repl_f, cv2.COLOR_LAB2BGR)
        
        # We only want to apply this adjustment INSIDE the mask
        return self.feather_boundary_blend(replacement, matched_bgr, bin_mask, 3)

    def seamless_clone_blend(self, original: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Alias for public access to seamlessClone fallback logic."""
        res = self._seamless_clone(original, replacement, mask)
        if res is not None:
            return res
        return replacement  # fallback if it fails

    def safe_blend(self, original: np.ndarray, replacement: np.ndarray, mask: np.ndarray, color_match: bool = True, feather_pixels: int = 12) -> np.ndarray:
        """
        A safe composite blending function for AI inpainting crops:
        1. Optionally color match.
        2. Try Poisson seamless blend.
        3. Fall back to feathered blend.
        """
        original, replacement, mask = self._normalize_blend_inputs(original, replacement, mask)
        result = replacement.copy()
        
        if color_match:
            result = self.match_color_to_surroundings(original, result, mask)
            
        # Try seamless clone
        sc_result = self._seamless_clone(original, result, mask)
        if sc_result is not None:
            return sc_result
            
        # Fallback to feathering
        return self.feather_boundary_blend(original, result, mask, feather_pixels)
