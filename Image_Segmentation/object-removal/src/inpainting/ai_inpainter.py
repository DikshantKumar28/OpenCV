"""
src/inpainting/ai_inpainter.py
------------------------------
Interface for AI inpainting backends.
"""
import numpy as np
import logging
from src.inpainting.lama_inpainter import LamaInpainter

logger = logging.getLogger(__name__)

class AIInpainter:
    def __init__(self, backend="lama", device="cpu"):
        self.backend = backend
        self.device = device
        self._inpainter = None
        
        if self.backend == "lama":
            self._inpainter = LamaInpainter(device=self.device)
        elif self.backend == "stable_diffusion":
            logger.warning("Stable Diffusion backend selected but not fully implemented. Falling back to LaMa if available.")
            self._inpainter = LamaInpainter(device=self.device)
        else:
            logger.warning(f"Unknown backend {self.backend}. Falling back to LaMa.")
            self._inpainter = LamaInpainter(device=self.device)

    def is_available(self) -> bool:
        """Returns True if the requested backend is successfully loaded and available."""
        if self._inpainter:
            return self._inpainter.is_available()
        return False

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the mask on the given image.
        Expects image and mask to be already cropped to the required region.
        image: BGR uint8
        mask: binary uint8 (255 = remove)
        Returns: BGR uint8
        """
        if not self.is_available():
            raise RuntimeError("AI inpainting backend is unavailable.")
            
        return self._inpainter.inpaint(image, mask)
