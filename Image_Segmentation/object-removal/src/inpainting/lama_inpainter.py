"""
src/inpainting/lama_inpainter.py
--------------------------------
LaMa inpainting backend using simple_lama_inpainting.
If simple_lama_inpainting is not installed, it gracefully disables itself.
"""
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class LamaInpainter:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self._available = False
        self._lazy_loaded = False

    def _lazy_load(self):
        if self._lazy_loaded:
            return
            
        self._lazy_loaded = True
        try:
            from simple_lama_inpainting import SimpleLama
            self.model = SimpleLama(device=self.device)
            self._available = True
            logger.info(f"LaMa inpainter loaded successfully on {self.device}.")
        except ImportError:
            logger.warning("simple_lama_inpainting is not installed. LaMa inpainting unavailable. "
                           "To install: pip install simple-lama-inpainting")
            self._available = False
        except Exception as e:
            logger.error(f"Error loading LaMa model: {e}")
            self._available = False

    def is_available(self) -> bool:
        if not self._lazy_loaded:
            self._lazy_load()
        return self._available

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("LaMa model is not available.")
            
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Ensure mask is single channel and convert to PIL
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # binarize to be safe
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        pil_mask = Image.fromarray(mask_bin).convert('L')
        
        try:
            result_pil = self.model(pil_image, pil_mask)
            result_rgb = np.array(result_pil)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            return result_bgr
        except Exception as e:
            logger.error(f"LaMa inference failed: {e}")
            # return original image if inference completely fails
            return image
