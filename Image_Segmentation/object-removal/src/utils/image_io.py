"""
src/utils/image_io.py
---------------------
Helpers for loading, saving, and validating images.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import uuid

# Import allowed extensions from config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ALLOWED_EXTENSIONS


def is_allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed list."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(original_name: str) -> str:
    """Generate a UUID-based filename preserving the original extension."""
    ext = Path(original_name).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def load_image_cv2(path: str | Path) -> np.ndarray:
    """
    Load an image from disk as a BGR NumPy array (OpenCV format).
    Raises FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load image as an RGB NumPy array."""
    bgr = load_image_cv2(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_cv2(img: np.ndarray, path: str | Path) -> Path:
    """
    Save a BGR NumPy array to disk.
    Creates parent directories if they don't exist.
    Returns the resolved Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), img)
    if not success:
        raise IOError(f"OpenCV failed to write image: {path}")
    return path


def save_image_pil(img_array: np.ndarray, path: str | Path) -> Path:
    """
    Save an RGB NumPy array using Pillow (useful for PNG with transparency).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(img_array)
    pil_img.save(str(path))
    return path


def numpy_to_pil(img: np.ndarray, mode: str = "RGB") -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    return Image.fromarray(img, mode)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array."""
    return np.array(img)


def get_image_dimensions(path: str | Path) -> tuple[int, int]:
    """
    Return (height, width) of an image without loading the full array.
    Uses PIL for speed.
    """
    with Image.open(str(path)) as img:
        w, h = img.size
    return h, w


def bytes_to_cv2(data: bytes) -> np.ndarray:
    """Decode raw image bytes (from file upload) into a BGR NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img


def cv2_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a BGR NumPy array to bytes in the given format."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"cv2.imencode failed for extension {ext}")
    return buf.tobytes()
