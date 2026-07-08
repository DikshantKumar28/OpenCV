"""
src/removal/multi_image_remover.py
------------------------------------
Removes selected objects from a target image using multiple
reference/duplicate images.

Key improvements:
1. Hard replacement (no blur over full region).
2. Boundary-only feathering with small seam cleanup.
3. Reference quality validation before use.
4. Smart fallback to OpenCV/AI based on object size and reference quality.

Pipeline:
1. Align references to the target using ORB/SIFT homography.
2. Build a median background from aligned references only (not target).
3. Validate reference quality by comparing boundary pixels only.
4. If valid: hard replace selected pixels, feather boundary, seam cleanup.
5. If invalid or large object: fallback to OpenCV or recommend AI.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    INPAINT_RADIUS,
    MASK_DILATE_KERNEL,
    MASK_DILATE_ITERATIONS,
    OUTPUT_DIR,
    MULTI_SMALL_MASK_RATIO,
    MULTI_MEDIUM_MASK_RATIO,
    MULTI_LARGE_MASK_RATIO,
    MULTI_MAX_FEATHER_RADIUS,
    MULTI_SEAM_INPAINT_RADIUS,
    REFERENCE_MIN_VALID_COUNT,
    REFERENCE_QUALITY_THRESHOLD,
    AI_INPAINTING_AVAILABLE,
    AI_INPAINTING_ENABLED,
)
from src.inpainting.ai_inpainter import AIInpainter
from src.alignment.image_aligner import ImageAligner
from src.removal.single_image_remover import SingleImageRemover
from src.utils.mask_utils import (
    combine_masks,
    expand_and_refine_mask,
    binarize_mask,
    get_mask_area_ratio,
    extract_boundary,
)
from src.utils.image_io import save_image_cv2

logger = logging.getLogger(__name__)


class MultiImageRemover:
    """
    Removes objects using aligned reference images for background reconstruction.

    Usage:
        remover = MultiImageRemover()
        result, mask, report = remover.remove(target, references, selected_masks)
    """

    def __init__(self):
        self.aligner = ImageAligner()
        self.single = SingleImageRemover()
        self.ai_inpainter = self.single.ai_inpainter if hasattr(self.single, 'ai_inpainter') else None

    def remove(
        self,
        target: np.ndarray,
        references: list[np.ndarray],
        masks: list[np.ndarray],
        dilate_kernel: int = MASK_DILATE_KERNEL,
        dilate_iterations: int = MASK_DILATE_ITERATIONS,
        inpaint_radius: int = INPAINT_RADIUS,
        removal_mode: str = "auto",
        use_strong_inpaint: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Remove selected objects using reference images.

        Returns:
            (result_image, combined_mask, report_dict)
        """
        if not masks:
            raise ValueError("No masks provided for removal.")

        combined_mask = combine_masks(masks)
        if np.count_nonzero(combined_mask > 127) == 0:
            raise ValueError("Selected object masks are empty.")

        mask_area_ratio = get_mask_area_ratio(combined_mask, target.shape)
        logger.info(
            f"Multi-image removal: mask_area_ratio={mask_area_ratio:.4f}, image_shape={target.shape}"
        )

        save_image_cv2(target, OUTPUT_DIR / "debug_multi_target.jpg")
        save_image_cv2(combined_mask, OUTPUT_DIR / "debug_multi_selected_mask.png")

        report = {
            "method_used": "",
            "mask_area_ratio": float(mask_area_ratio),
            "valid_reference_count": 0,
            "reference_quality_score": 0.0,
            "fallback_used": False,
            "warning": "",
            "mask_white_pixels": int(np.count_nonzero(combined_mask > 127)),
            "image_shape": list(target.shape),
        }

        explicit_mode = removal_mode in ["opencv_fast", "opencv_strong", "ai_inpainting", "reference_replacement"]
        
        # ────────────────────────────────────────────────────────────
        # Explicit user mode: respect the choice
        # ────────────────────────────────────────────────────────────
        if removal_mode == "opencv_fast":
            report["method_used"] = "opencv_fast_forced"
            report["warning"] = "OpenCV fast inpainting may create artifacts for large objects."
            return self._fallback_opencv(
                target, masks, dilate_kernel, dilate_iterations, inpaint_radius, False, report
            )

        if removal_mode == "opencv_strong":
            report["method_used"] = "opencv_strong_forced"
            return self._fallback_opencv(
                target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report
            )

        if removal_mode == "ai_inpainting" or removal_mode == "ai_eraser":
            report["method_used"] = "ai_required"
            return self._fallback_opencv(
                target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report, override_mode="ai_inpainting"
            )

        # ────────────────────────────────────────────────────────────
        # Explicit reference replacement
        # ────────────────────────────────────────────────────────────
        if removal_mode == "reference_replacement":
            if not references:
                report["warning"] = "No reference images provided for reference replacement."
                return self._fallback_opencv(
                    target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report
                )
            return self._do_reference_replacement(
                target,
                references,
                combined_mask,
                dilate_kernel,
                dilate_iterations,
                inpaint_radius,
                report,
                force_use=True,
            )

        # ────────────────────────────────────────────────────────────
        # Auto mode: intelligent method selection
        # ────────────────────────────────────────────────────────────
        if removal_mode == "auto":
            # Case A: Small objects -> use OpenCV inpaint
            if mask_area_ratio <= MULTI_SMALL_MASK_RATIO:
                logger.info("Auto mode: small object -> OpenCV strong inpaint")
                report["method_used"] = "opencv_strong"
                return self._fallback_opencv(
                    target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report
                )

            # Case B: Medium objects -> try reference, fallback to OpenCV
            if mask_area_ratio <= MULTI_MEDIUM_MASK_RATIO:
                logger.info("Auto mode: medium object -> try reference replacement")
                if references:
                    result, mask, ref_report = self._do_reference_replacement(
                        target,
                        references,
                        combined_mask,
                        dilate_kernel,
                        dilate_iterations,
                        inpaint_radius,
                        report,
                        force_use=False,
                    )
                    if ref_report["method_used"] == "reference_replacement":
                        return result, mask, ref_report
                    # Reference failed; fallback to OpenCV
                logger.info("Auto mode: reference failed or unavailable -> OpenCV strong fallback")
                return self._fallback_opencv(
                    target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report
                )

            # Case C: Large objects -> require AI or very high-quality reference
            logger.info("Auto mode: large object -> AI recommended or high-quality reference required")
            if references:
                result, mask, ref_report = self._do_reference_replacement(
                    target,
                    references,
                    combined_mask,
                    dilate_kernel,
                    dilate_iterations,
                    inpaint_radius,
                    report,
                    force_use=False,
                )
                if ref_report["method_used"] == "reference_replacement":
                    return result, mask, ref_report
            
            report["method_used"] = "ai_required"
            report["warning"] = "Large selected object requires AI inpainting or a clean, well-aligned reference image."
            logger.warning(f"Auto mode: AI required (mask_area={mask_area_ratio:.4f})")
            return self._fallback_opencv(
                target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report, override_mode="ai_inpainting" if AI_INPAINTING_ENABLED else "opencv_strong"
            )

        # Fallback
        report["method_used"] = "opencv_strong"
        return self._fallback_opencv(
            target, masks, dilate_kernel, dilate_iterations, inpaint_radius, True, report, override_mode=removal_mode
        )

    def _do_reference_replacement(
        self,
        target,
        references,
        combined_mask,
        dilate_kernel,
        dilate_iterations,
        inpaint_radius,
        report,
        force_use=False,
    ):
        """
        Attempt reference-based replacement.
        Returns (result, mask, report) where method_used indicates success or failure.
        """
        if not references:
            report["method_used"] = "reference_unavailable"
            report["warning"] = "No reference images provided."
            return self._fallback_opencv(
                target, [combined_mask], dilate_kernel, dilate_iterations, inpaint_radius, True, report
            )

        logger.info(f"Aligning {len(references)} reference image(s)…")
        aligned_refs = self.aligner.align_multiple(references, target)
        report["valid_reference_count"] = len(aligned_refs)

        if len(aligned_refs) < REFERENCE_MIN_VALID_COUNT:
            logger.warning(f"Not enough valid references: {len(aligned_refs)}")
            report["method_used"] = "reference_alignment_failed"
            report["warning"] = "Reference image alignment failed. Not enough valid references."
            return self._fallback_opencv(
                target, [combined_mask], dilate_kernel, dilate_iterations, inpaint_radius, True, report
            )

        # Save aligned references for debug
        for i, ref in enumerate(aligned_refs):
            save_image_cv2(ref, OUTPUT_DIR / f"debug_multi_aligned_reference_{i}.jpg")

        # Build background from aligned references only (not target)
        background = self._build_median_background(aligned_refs)
        save_image_cv2(background, OUTPUT_DIR / "debug_multi_background.jpg")

        # Expand and refine the mask
        inpaint_mask, _ = expand_and_refine_mask(
            combined_mask,
            target.shape,
            kernel_size=dilate_kernel,
            iterations=dilate_iterations,
        )
        save_image_cv2(inpaint_mask, OUTPUT_DIR / "debug_multi_expanded_mask.png")

        # Validate reference quality
        quality_score = self._evaluate_reference_quality(
            target,
            background,
            combined_mask,
        )
        report["reference_quality_score"] = float(quality_score)

        threshold = REFERENCE_QUALITY_THRESHOLD
        if not force_use and quality_score < threshold:
            logger.warning(f"Reference quality score too low: {quality_score:.3f} < {threshold:.3f}")
            report["method_used"] = "reference_poor_quality"
            report["warning"] = (
                f"Reference background quality is low ({quality_score:.2f}). "
                "Falling back to AI inpainting if available."
            )
            return self._fallback_opencv(
                target, [combined_mask], dilate_kernel, dilate_iterations, inpaint_radius, True, report, override_mode="ai_inpainting" if AI_INPAINTING_ENABLED else "opencv_strong"
            )

        # Perform hard replacement + boundary feathering
        result, boundary_mask = self._hard_replace_with_feather(
            target,
            background,
            inpaint_mask,
            combined_mask,
            dilate_kernel,
            inpaint_radius,
        )

        # Optional AI seam cleanup
        ai_cleanup_used = False
        if AI_INPAINTING_ENABLED and self.ai_inpainter and self.ai_inpainter.is_available():
            try:
                # We do a quick AI pass over the boundary seam if it is thick enough
                if np.count_nonzero(boundary_mask) > 50:
                    result = self.ai_inpainter.inpaint(result, boundary_mask)
                    ai_cleanup_used = True
            except Exception as e:
                logger.warning(f"AI seam cleanup failed: {e}")

        save_image_cv2(result, OUTPUT_DIR / "debug_multi_final.jpg")

        if ai_cleanup_used:
            report["method_used"] = "reference_plus_ai_cleanup"
        else:
            report["method_used"] = "reference_replacement"
            
        report["fallback_used"] = False

        # Save debug report
        report_path = OUTPUT_DIR / "debug_multi_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        return result, combined_mask, report

    def _hard_replace_with_feather(
        self,
        target,
        background,
        inpaint_mask,
        combined_mask,
        dilate_kernel,
        inpaint_radius,
    ):
        """
        Perform hard replacement (no blur over full region).

        Steps:
        1. Binary replacement: result = target, then replace masked pixels with background.
        2. Extract boundary mask (thin band around object edge).
        3. Feather boundary with small blur.
        4. Seam cleanup with small inpaint radius on boundary only.
        """
        binary_mask = binarize_mask(inpaint_mask)

        # Step 1: Hard replacement
        result = target.copy()
        result[binary_mask > 127] = background[binary_mask > 127]

        # Step 2: Extract boundary mask (thin band)
        # Erode to get interior, dilate to get exterior, boundary = exterior - interior
        erode_kernel = max(3, dilate_kernel - 2)
        eroded = cv2.erode(
            binary_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel)),
            iterations=1,
        )
        dilated = cv2.dilate(
            binary_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)),
            iterations=1,
        )
        boundary_mask = cv2.subtract(dilated, eroded)
        save_image_cv2(boundary_mask, OUTPUT_DIR / "debug_multi_boundary_mask.png")

        # Step 3: Small feathering blur on boundary
        # Only feather the boundary pixels to smooth the seam
        feather_radius = min(MULTI_MAX_FEATHER_RADIUS, max(5, dilate_kernel - 2))
        # Use odd kernel size
        feather_kernel = max(5, feather_radius * 2 - 1)
        if feather_kernel % 2 == 0:
            feather_kernel += 1

        # Apply small Gaussian blur only to boundary pixels
        boundary_blurred = cv2.GaussianBlur(result, (feather_kernel, feather_kernel), 0)
        
        # Blend: use blurred where boundary_mask > 0, else use hard result
        boundary_mask_norm = boundary_mask.astype(np.float32) / 255.0
        if len(result.shape) == 3:
            boundary_mask_norm = boundary_mask_norm[:, :, np.newaxis]
        
        result = (result.astype(np.float32) * (1.0 - boundary_mask_norm) + 
                  boundary_blurred.astype(np.float32) * boundary_mask_norm).astype(np.uint8)

        # Step 4: Seam cleanup inpaint on boundary only (small radius)
        try:
            seam_radius = MULTI_SEAM_INPAINT_RADIUS
            if seam_radius > 0 and np.count_nonzero(boundary_mask) > 0:
                result = cv2.inpaint(
                    result,
                    boundary_mask,
                    seam_radius,
                    cv2.INPAINT_TELEA,
                )
        except cv2.error as e:
            logger.warning(f"Seam cleanup inpaint failed: {e}")

        return result, boundary_mask

    def _evaluate_reference_quality(self, target, background, mask):
        """
        Evaluate reference quality by comparing boundary pixels only.

        Returns:
            quality_score: float in [0, 1] where 1 is perfect match.
        """
        binary_mask = binarize_mask(mask)
        y_idxs, x_idxs = np.where(binary_mask > 127)
        
        if len(x_idxs) == 0 or len(y_idxs) == 0:
            return 0.0

        # Get bounding box of mask
        x_min, x_max = x_idxs.min(), x_idxs.max()
        y_min, y_max = y_idxs.min(), y_idxs.max()

        # Expand slightly to include boundary area
        pad = 15
        x1 = max(0, x_min - pad)
        x2 = min(target.shape[1], x_max + pad)
        y1 = max(0, y_min - pad)
        y2 = min(target.shape[0], y_max + pad)

        # Extract patches
        target_patch = target[y1:y2, x1:x2]
        background_patch = background[y1:y2, x1:x2]
        mask_patch = binary_mask[y1:y2, x1:x2]

        if target_patch.size == 0 or background_patch.size == 0:
            return 0.0

        # Evaluate only boundary pixels (not deep inside the object)
        # Create a small region around the boundary
        erode_kernel = 5
        eroded_patch = cv2.erode(
            mask_patch,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel)),
            iterations=1,
        )
        boundary_patch = mask_patch - eroded_patch

        if np.count_nonzero(boundary_patch) < 10:
            # Not enough boundary pixels, evaluate full patch
            boundary_region = mask_patch > 0
        else:
            boundary_region = boundary_patch > 0

        if not np.any(boundary_region):
            # No region to evaluate
            return 0.5

        # Compare only the boundary region
        target_gray = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        background_gray = cv2.cvtColor(background_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

        brightness_diff = float(
            np.mean(np.abs(target_gray[boundary_region] - background_gray[boundary_region]))
        )
        color_diff = float(
            np.mean(np.abs(target_patch[boundary_region].astype(np.float32) - 
                          background_patch[boundary_region].astype(np.float32)))
        )

        # Penalize large brightness differences
        norm_brightness = min(1.0, brightness_diff / 50.0)
        norm_color = min(1.0, color_diff / 50.0)

        quality_score = 1.0 - (norm_brightness * 0.6 + norm_color * 0.4)
        quality_score = float(max(0.0, min(1.0, quality_score)))

        logger.info(
            f"Reference boundary quality | brightness_diff={brightness_diff:.1f} | color_diff={color_diff:.1f} | score={quality_score:.3f}"
        )

        return quality_score

    def _fallback_opencv(
        self,
        target,
        masks,
        dilate_kernel,
        dilate_iterations,
        inpaint_radius,
        use_strong_inpaint,
        report,
        override_mode=None
    ):
        """
        Fallback to single-image OpenCV inpainting.
        """
        logger.info("Falling back to single-image inpainting")
        
        mode = override_mode if override_mode else ("opencv_strong" if use_strong_inpaint else "opencv_fast")
        
        result, combined_mask, single_report = self.single.remove(
            target,
            masks,
            dilate_kernel=dilate_kernel,
            dilate_iterations=dilate_iterations,
            inpaint_radius=inpaint_radius,
            removal_mode=mode,
            use_strong_inpaint=use_strong_inpaint,
        )

        # Merge reports
        if not report.get("method_used"):
            report["method_used"] = single_report.get("method_used", "opencv_strong_fallback")
        report["fallback_used"] = True
        report["mask_area_ratio"] = float(single_report.get("mask_area_ratio", 0.0))

        # Save debug
        report_path = OUTPUT_DIR / "debug_multi_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        return result, combined_mask, report

    def _build_median_background(self, aligned_refs: list[np.ndarray]) -> np.ndarray:
        """Build median background from aligned references only (not target)."""
        if not aligned_refs:
            raise ValueError("No aligned references provided")
        h, w = aligned_refs[0].shape[:2]
        stack = np.stack([ref.astype(np.float32) for ref in aligned_refs], axis=0)
        median_bg = np.median(stack, axis=0).astype(np.uint8)
        return median_bg

