"""
tests/test_removal.py
----------------------
Unit tests for SingleImageRemover and MultiImageRemover.
Uses synthetic images — no YOLO/SAM models needed.
"""

import numpy as np
import cv2
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.removal.single_image_remover import SingleImageRemover
from src.removal.multi_image_remover import MultiImageRemover
from src.utils.mask_utils import bbox_to_mask


def _solid_image(h=200, w=300, color=(120, 180, 60)) -> np.ndarray:
    """Create a solid-colour BGR image."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def _image_with_rectangle(h=200, w=300) -> np.ndarray:
    """Create an image with a bright red rectangle (simulates an object)."""
    img = _solid_image(h, w, (50, 50, 50))
    cv2.rectangle(img, (80, 60), (160, 130), (0, 0, 255), -1)
    return img


class TestSingleImageRemover:

    def setup_method(self):
        self.remover = SingleImageRemover()

    def test_remove_returns_same_shape(self):
        image = _image_with_rectangle()
        mask  = bbox_to_mask([80, 60, 160, 130], 200, 300)
        result, combined, _ = self.remover.remove(image, [mask])
        assert result.shape == image.shape

    def test_remove_output_dtype(self):
        image = _image_with_rectangle()
        mask  = bbox_to_mask([80, 60, 160, 130], 200, 300)
        result, _, _ = self.remover.remove(image, [mask])
        assert result.dtype == np.uint8

    def test_remove_multiple_masks(self):
        image = _image_with_rectangle()
        m1 = bbox_to_mask([80, 60, 120, 100], 200, 300)
        m2 = bbox_to_mask([120, 100, 160, 130], 200, 300)
        result, combined, _ = self.remover.remove(image, [m1, m2])
        assert result.shape == image.shape
        # combined mask should be union of m1 and m2
        assert combined[90, 100] == 255
        assert combined[115, 115] == 255

    def test_remove_empty_masks_raises(self):
        image = _image_with_rectangle()
        with pytest.raises(ValueError):
            self.remover.remove(image, [])

    def test_combined_mask_returned(self):
        image = _image_with_rectangle()
        mask  = bbox_to_mask([80, 60, 160, 130], 200, 300)
        _, combined, _ = self.remover.remove(image, [mask])
        assert combined.shape == (200, 300)
        assert combined.dtype == np.uint8


class TestMultiImageRemover:

    def setup_method(self):
        self.remover = MultiImageRemover()

    def test_fallback_to_single_when_no_refs(self):
        """With no references, should fall back to single-image inpainting."""
        image = _image_with_rectangle()
        mask  = bbox_to_mask([80, 60, 160, 130], 200, 300)
        result, combined, _ = self.remover.remove(image, [], [mask])
        assert result.shape == image.shape

    def test_with_identical_references(self):
        """Using identical images as references should produce clean background."""
        # Background image (what we want to recover)
        bg = _solid_image(200, 300, (80, 120, 200))
        # Target = background + a red box (the "object")
        target = bg.copy()
        cv2.rectangle(target, (80, 60), (160, 130), (0, 0, 255), -1)

        # References = clean background (same as bg)
        refs = [bg.copy(), bg.copy()]
        mask = bbox_to_mask([80, 60, 160, 130], 200, 300)

        result, _, _ = self.remover.remove(target, refs, [mask])
        assert result.shape == target.shape

    def test_all_blank_refs_fallback(self):
        """All-blank references will fail alignment → single-image fallback."""
        image = _image_with_rectangle()
        blank = np.zeros_like(image)
        mask  = bbox_to_mask([80, 60, 160, 130], 200, 300)
        result, _, _ = self.remover.remove(image, [blank, blank], [mask])
        assert result.shape == image.shape

    def test_empty_masks_raises(self):
        image = _image_with_rectangle()
        with pytest.raises(ValueError):
            self.remover.remove(image, [], [])
