"""
tests/test_detection.py
------------------------
Unit tests for YoloDetector and mask_utils detection helpers.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import (
    create_empty_mask, combine_masks, resize_mask,
    dilate_mask, bbox_to_mask, polygon_to_mask, normalize_yolo_mask,
)


# ── mask_utils tests ─────────────────────────────────────────────────

def test_create_empty_mask():
    mask = create_empty_mask(100, 200)
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert mask.sum() == 0


def test_bbox_to_mask():
    mask = bbox_to_mask([10, 10, 50, 50], 100, 100)
    assert mask.shape == (100, 100)
    assert mask[10:50, 10:50].min() == 255
    assert mask[0, 0] == 0
    assert mask[60, 60] == 0


def test_combine_masks_union():
    m1 = bbox_to_mask([0, 0, 20, 20], 100, 100)
    m2 = bbox_to_mask([50, 50, 80, 80], 100, 100)
    combined = combine_masks([m1, m2])
    # Region 1 should be filled
    assert combined[10, 10] == 255
    # Region 2 should be filled
    assert combined[60, 60] == 255
    # Untouched area should be black
    assert combined[30, 30] == 0


def test_combine_masks_single():
    m = bbox_to_mask([5, 5, 15, 15], 50, 50)
    combined = combine_masks([m])
    assert np.array_equal(combined, m)


def test_combine_masks_empty_raises():
    with pytest.raises(ValueError):
        combine_masks([])


def test_resize_mask():
    mask = bbox_to_mask([0, 0, 40, 40], 80, 80)
    resized = resize_mask(mask, 160, 160)
    assert resized.shape == (160, 160)
    # Scaled bbox should be ~0..80 in the new resolution
    assert resized[5, 5] == 255
    assert resized[150, 150] == 0


def test_dilate_mask_expands():
    mask = create_empty_mask(100, 100)
    mask[50, 50] = 255               # single pixel
    dilated = dilate_mask(mask, kernel_size=11)
    # Should have expanded around the single pixel
    assert dilated.sum() > mask.sum()


def test_polygon_to_mask():
    polygon = [[10, 10], [40, 10], [40, 40], [10, 40]]
    mask = polygon_to_mask(polygon, 100, 100)
    assert mask[25, 25] == 255       # inside
    assert mask[5, 5]   == 0         # outside


def test_normalize_yolo_mask():
    raw = np.ones((160, 160), dtype=np.float32) * 0.9
    norm = normalize_yolo_mask(raw, 480, 640)
    assert norm.shape == (480, 640)
    assert norm.dtype == np.uint8
    assert norm[0, 0] == 255
