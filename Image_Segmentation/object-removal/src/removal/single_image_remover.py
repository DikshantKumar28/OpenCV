"""
src/removal/single_image_remover.py
-------------------------------------
Removes selected objects from a single image using OpenCV inpainting.

Pipeline:
1. Combine selected masks.
2. Expand/refine the mask to cover object boundaries.
3. Choose strong or fast inpaint strategy automatically.
4. Blend only the boundary back into the original image.
5. Save debug artifacts and method metadata.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MASK_DILATE_KERNEL,
    MASK_DILATE_ITERATIONS,
    INPAINT_RADIUS,
    OUTPUT_DIR,
    SMALL_MASK_AREA_RATIO,
    MEDIUM_MASK_AREA_RATIO,
    LARGE_MASK_AREA_RATIO,
    AI_INPAINTING_AVAILABLE,
    AI_INPAINTING_ENABLED,
    AI_INPAINTING_BACKEND,
    AI_INPAINTING_DEVICE,
    AI_INPAINTING_MIN_MASK_RATIO,
    MASK_EXPAND_PIXELS_AI,
    MASK_FEATHER_PIXELS,
    COLOR_MATCH_ENABLED,
    AI_INPAINTING_PADDING,
)
from src.inpainting.ai_inpainter import AIInpainter
from src.utils.mask_utils import (
    combine_masks,
    expand_and_refine_mask,
    binarize_mask,
    get_mask_area_ratio,
    extract_boundary,
    prepare_ai_inpaint_mask,
    crop_to_mask_bbox,
    paste_crop_back,
)
from src.blending.blender import Blender
from src.utils.image_io import save_image_cv2

logger = logging.getLogger(__name__)


class SingleImageRemover:
    """
    Removes objects from a single image using OpenCV inpainting.

    Usage:
        remover = SingleImageRemover()
        result, mask, report = remover.remove(image, [mask1, mask2])
    """

    def __init__(self):
        self.blender = Blender()
        self.ai_inpainter = AIInpainter(backend=AI_INPAINTING_BACKEND, device=AI_INPAINTING_DEVICE) if AI_INPAINTING_ENABLED else None

    def remove(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        dilate_kernel: int = MASK_DILATE_KERNEL,
        dilate_iterations: int = MASK_DILATE_ITERATIONS,
        inpaint_radius: int = INPAINT_RADIUS,
        removal_mode: str = "auto",
        use_strong_inpaint: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        if not masks:
            raise ValueError("No masks provided for removal.")

        combined_mask = combine_masks(masks)
        if np.count_nonzero(combined_mask > 127) == 0:
            raise ValueError("Selected object masks are empty.")

        mask_area_ratio = get_mask_area_ratio(combined_mask, image.shape)
        selected_pixel_count = int(np.count_nonzero(combined_mask > 127))
        logger.info(
            f"Single-image removal: mask_area_ratio={mask_area_ratio:.4f}, white_pixels={selected_pixel_count}, image_shape={image.shape}, mask_shape={combined_mask.shape}"
        )

        save_image_cv2(image, OUTPUT_DIR / "debug_single_original.jpg")
        save_image_cv2(combined_mask, OUTPUT_DIR / "debug_single_selected_mask.png")

        ai_available = (
            AI_INPAINTING_AVAILABLE
            and self.ai_inpainter is not None
            and self.ai_inpainter.is_available()
        )

        method_used, warning, use_fast = self._choose_method(removal_mode, mask_area_ratio, use_strong_inpaint, ai_available)

        report = {
            "method_used": method_used,
            "mask_area_ratio": float(mask_area_ratio),
            "mean_difference_inside_mask": 0.0,
            "warning": warning,
            "selected_pixel_count": selected_pixel_count,
            "ai_available": ai_available,
            "debug": {}
        }

        result = None
        current_dilate_iters = dilate_iterations
        current_inpaint_radius = inpaint_radius
        max_retries = 1

        if method_used.startswith("ai_"):
            try:
                # 1. Prepare mask
                expanded_mask, blurred_mask = prepare_ai_inpaint_mask(combined_mask, image.shape, MASK_EXPAND_PIXELS_AI)
                save_image_cv2(expanded_mask, OUTPUT_DIR / "debug_ai_expanded_mask.png")
                
                # 2. Crop
                img_crop, mask_crop, bbox = crop_to_mask_bbox(image, expanded_mask, AI_INPAINTING_PADDING)
                save_image_cv2(img_crop, OUTPUT_DIR / "debug_ai_crop.jpg")
                save_image_cv2(mask_crop, OUTPUT_DIR / "debug_ai_crop_mask.png")
                
                # 3. Inpaint
                inpainted_crop = self.ai_inpainter.inpaint(img_crop, mask_crop)
                save_image_cv2(inpainted_crop, OUTPUT_DIR / "debug_ai_inpainted_crop.jpg")
                
                # 4. Blend locally
                x1, y1, x2, y2 = bbox
                blur_mask_crop = blurred_mask[y1:y2, x1:x2]
                blended_crop = self.blender.safe_blend(img_crop, inpainted_crop, blur_mask_crop, COLOR_MATCH_ENABLED, MASK_FEATHER_PIXELS)
                
                # 5. Paste back
                result = paste_crop_back(image, blended_crop, bbox)
                save_image_cv2(result, OUTPUT_DIR / "debug_ai_blended_full.jpg")
                
                report["debug"] = {
                    "mask_pixels": selected_pixel_count,
                    "bbox": bbox,
                    "crop_used": True
                }
            except Exception as e:
                logger.error(f"AI Inpainting failed: {e}")
                report["warning"] = f"AI Inpainting failed ({e}). Falling back to OpenCV."
                report["method_used"] = "opencv_strong_fallback"
                method_used = "opencv_strong"
                use_fast = False

        if not method_used.startswith("ai_"):
            for attempt in range(max_retries + 1):
                logger.info(
                    f"Single-image inpainting attempt {attempt + 1} | method={method_used} | dilate_iters={current_dilate_iters} | radius={current_inpaint_radius}"
                )

                inpaint_mask, blurred_mask = expand_and_refine_mask(
                    combined_mask,
                    image.shape,
                    kernel_size=dilate_kernel,
                    iterations=current_dilate_iters,
                )
                save_image_cv2(inpaint_mask, OUTPUT_DIR / "debug_single_expanded_mask.png")

                telea = cv2.inpaint(image, inpaint_mask, current_inpaint_radius, cv2.INPAINT_TELEA)
                save_image_cv2(telea, OUTPUT_DIR / "debug_single_telea.jpg")
                ns = None
                if not use_fast:
                    ns = cv2.inpaint(image, inpaint_mask, current_inpaint_radius, cv2.INPAINT_NS)
                    save_image_cv2(ns, OUTPUT_DIR / "debug_single_ns.jpg")
                    inpainted = cv2.addWeighted(telea, 0.75, ns, 0.25, 0)
                else:
                    inpainted = telea
                    # Save a copy in case debugging expects both files
                    save_image_cv2(inpainted, OUTPUT_DIR / "debug_single_ns.jpg")

                blur_radius = max(7, min(21, dilate_kernel * 2 + 1))
                result = self.blender.feathered_blend(image, inpainted, blurred_mask, blur_radius=blur_radius)

                original_mask = (combined_mask > 127)
                if np.any(original_mask):
                    diff = np.abs(image[original_mask].astype(np.float32) - result[original_mask].astype(np.float32))
                    mean_diff = float(np.mean(diff))
                    report["mean_difference_inside_mask"] = mean_diff
                    logger.info(f"Mean pixel difference inside selected mask: {mean_diff:.2f}")

                    if mean_diff < 8.0 and attempt < max_retries:
                        logger.warning(
                            "Removal was too subtle. Increasing dilation and radius for one retry."
                        )
                        current_dilate_iters = min(current_dilate_iters + 2, 12)
                        current_inpaint_radius = min(current_inpaint_radius + 5, 35)
                        continue

                break

        save_image_cv2(result, OUTPUT_DIR / "debug_single_final.jpg")
        save_image_cv2(blurred_mask, OUTPUT_DIR / "debug_single_selected_mask.png")

        report_path = OUTPUT_DIR / "debug_single_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        return result, combined_mask, report

    def _choose_method(self, removal_mode: str, mask_area_ratio: float, use_strong_inpaint: bool, ai_available: bool) -> tuple[str, str, bool]:
        removal_mode = (removal_mode or "auto").lower()

        if removal_mode == "opencv_fast":
            return "opencv_fast", "OpenCV Fast inpainting selected.", True

        if removal_mode == "opencv_strong":
            return "opencv_strong", "OpenCV Strong inpainting selected.", False

        if removal_mode == "ai_eraser" or removal_mode == "ai_inpainting":
            if ai_available:
                return f"ai_{AI_INPAINTING_BACKEND}", "Using AI inpainting.", False
            return (
                "opencv_fast",
                "AI inpainting backend is not installed. Large object removal with OpenCV may look blurry.",
                True,
            )

        # Auto mode
        if mask_area_ratio <= AI_INPAINTING_MIN_MASK_RATIO:
            return (
                "opencv_strong",
                "Selected object is small. OpenCV Strong inpainting is used.",
                False,
            )

        # Large object
        if ai_available:
            return (
                f"ai_{AI_INPAINTING_BACKEND}",
                "Selected object is large. AI inpainting is used for realistic removal.",
                False,
            )

        return (
            "opencv_fast",
            "AI inpainting backend is not installed. Large object removal with OpenCV may look blurry.",
            True,
        )
