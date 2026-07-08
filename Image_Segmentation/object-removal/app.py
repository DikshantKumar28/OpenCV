"""
app.py
------
FastAPI server for the object-removal application.

Endpoints:
  GET  /api/health
  POST /api/detect              Upload image(s), run detection, return detections
  POST /api/remove-single       Remove objects from a single image
  POST /api/remove-multiple     Remove objects using reference images
  GET  /api/output/{filename}   Serve result images to the frontend
"""

import uuid
import json
import logging
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Project imports ──────────────────────────────────────────────────
from config import (
    INPUT_SINGLE, INPUT_MULTIPLE, OUTPUT_DIR, MASKS_DIR,
    ALLOWED_EXTENSIONS, CORS_ORIGINS, API_HOST, API_PORT,
)
from src.utils.image_io import is_allowed_file, generate_unique_filename, bytes_to_cv2, save_image_cv2
from src import pipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Object Removal API",
    description="Remove objects from images using YOLO + SAM + OpenCV inpainting.",
    version="1.0.0",
)

# ── CORS (allow Vite dev server) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve output images as static files ─────────────────────────────
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/masks",  StaticFiles(directory=str(MASKS_DIR)),  name="masks")

# ── In-memory detection cache ─────────────────────────────────────────
# Maps session_id → { "image_path": str, "detections": list[dict] }
# Detections include numpy 'mask' arrays which cannot be sent over JSON.
_session_cache: dict[str, dict] = {}


# ════════════════════════════════════════════════════════════════════════
# Helper: save uploaded file to disk
# ════════════════════════════════════════════════════════════════════════
async def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    if not is_allowed_file(upload.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed: {upload.filename}. "
                   f"Allowed: {ALLOWED_EXTENSIONS}",
        )
    data = await upload.read()
    unique_name = generate_unique_filename(upload.filename)
    save_path   = dest_dir / unique_name

    # Decode & re-save through OpenCV to standardise format
    img = bytes_to_cv2(data)
    save_image_cv2(img, save_path)
    return save_path


# ════════════════════════════════════════════════════════════════════════
# GET /api/health
# ════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Object Removal API is running."}


# ════════════════════════════════════════════════════════════════════════
# POST /api/detect
# Body: multipart/form-data
#   images: one or more uploaded image files
#           (first image is always the target; rest are references)
# ════════════════════════════════════════════════════════════════════════
@app.post("/api/detect")
async def detect(
    images: List[UploadFile] = File(...),
    conf: float = Form(0.15)
):
    """
    Upload one or more images and run object detection on the first (target) image.
    Returns detected objects with bounding boxes and base64 mask previews.
    Also returns a session_id used in subsequent removal calls.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    # Determine mode
    mode = "single" if len(images) == 1 else "multiple"
    dest = INPUT_SINGLE if mode == "single" else INPUT_MULTIPLE

    # Save all uploaded images
    saved_paths: list[Path] = []
    for upload in images:
        path = await _save_upload(upload, dest)
        saved_paths.append(path)

    target_path = saved_paths[0]
    ref_paths   = saved_paths[1:]

    # Run detection on target
    try:
        result = pipeline.detect_objects(target_path, conf=conf)
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    # Cache detections (including numpy masks) server-side
    session_id = uuid.uuid4().hex
    _session_cache[session_id] = {
        "mode":         mode,
        "target_path":  str(target_path),
        "ref_paths":    [str(p) for p in ref_paths],
        # The pipeline now returns the raw detections (with compressed masks) directly
        "detections":   result["raw_detections"],
    }

    return JSONResponse({
        "session_id":    session_id,
        "mode":          mode,
        "detections":    result["detections"],
        "preview_name":  result["preview_name"],
        "image_width":   _get_image_size(target_path)[1],
        "image_height":  _get_image_size(target_path)[0],
    })


# ════════════════════════════════════════════════════════════════════════
# POST /api/remove-single
# Body: JSON { session_id, selected_ids }
# ════════════════════════════════════════════════════════════════════════
@app.post("/api/remove-single")
async def remove_single(
    session_id:   str       = Form(...),
    selected_ids: str       = Form(...),   # JSON-encoded list of ints
    mask_dilate_iterations: int = Form(4),
    mask_dilate_kernel: int = Form(9),
    inpaint_radius: int = Form(15),
    use_strong_inpaint: bool = Form(True),
    removal_mode: str = Form("auto"),
):
    """
    Remove selected objects from the previously uploaded single image.
    Uses OpenCV inpainting.
    """
    session = _get_session(session_id)
    ids     = _parse_ids(selected_ids)

    try:
        result = pipeline.remove_from_single_image(
            image_path   = session["target_path"],
            detections   = session["detections"],
            selected_ids = ids,
            dilate_kernel=mask_dilate_kernel,
            dilate_iterations=mask_dilate_iterations,
            inpaint_radius=inpaint_radius,
            removal_mode=removal_mode,
            use_strong_inpaint=use_strong_inpaint,
        )
    except Exception as e:
        logger.exception("Single-image removal failed")
        raise HTTPException(status_code=500, detail=f"Removal error: {e}")

    return JSONResponse({
        "output_name": result["output_name"],
        "output_url":  f"/output/{result['output_name']}",
        **result.get("report", {}),
    })


# ════════════════════════════════════════════════════════════════════════
# POST /api/remove-multiple
# Body: JSON { session_id, selected_ids }
# ════════════════════════════════════════════════════════════════════════
@app.post("/api/remove-multiple")
async def remove_multiple(
    session_id:   str = Form(...),
    selected_ids: str = Form(...),
    mask_dilate_iterations: int = Form(4),
    mask_dilate_kernel: int = Form(9),
    inpaint_radius: int = Form(15),
    use_strong_inpaint: bool = Form(True),
    removal_mode: str = Form("auto"),
):
    """
    Remove selected objects using reference images for background reconstruction.
    If no references were uploaded or alignment fails, falls back to inpainting.
    """
    session = _get_session(session_id)
    ids     = _parse_ids(selected_ids)

    ref_paths = session.get("ref_paths", [])

    try:
        if ref_paths:
            result = pipeline.remove_from_multiple_images(
                target_path     = session["target_path"],
                reference_paths = ref_paths,
                detections      = session["detections"],
                selected_ids    = ids,
                dilate_kernel=mask_dilate_kernel,
                dilate_iterations=mask_dilate_iterations,
                inpaint_radius=inpaint_radius,
                removal_mode=removal_mode,
                use_strong_inpaint=use_strong_inpaint,
            )
        else:
            logger.warning("No reference images in session — using single-image fallback.")
            result = pipeline.remove_from_single_image(
                image_path   = session["target_path"],
                detections   = session["detections"],
                selected_ids = ids,
                dilate_kernel=mask_dilate_kernel,
                dilate_iterations=mask_dilate_iterations,
                inpaint_radius=inpaint_radius,
                removal_mode=removal_mode,
                use_strong_inpaint=use_strong_inpaint,
            )
    except Exception as e:
        logger.exception("Multi-image removal failed")
        raise HTTPException(status_code=500, detail=f"Removal error: {e}")

    response = {
        "output_name": result["output_name"],
        "output_url":  f"/output/{result['output_name']}",
    }
    response.update(result.get("report", {}))
    return JSONResponse(response)


# ════════════════════════════════════════════════════════════════════════
# GET /api/output/{filename}
# ════════════════════════════════════════════════════════════════════════
@app.get("/api/output/{filename}")
def get_output(filename: str):
    """Serve a saved output image by filename."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(file_path))


# ════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════
def _get_session(session_id: str) -> dict:
    session = _session_cache.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please re-upload your images."
        )
    return session


def _parse_ids(raw: str) -> list[int]:
    try:
        ids = json.loads(raw)
        if not isinstance(ids, list):
            raise ValueError
        return [int(i) for i in ids]
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"selected_ids must be a JSON array of integers, got: {raw!r}"
        )





def _get_image_size(path: Path) -> tuple[int, int]:
    """Return (height, width) of image at path."""
    img = cv2.imread(str(path))
    if img is None:
        return (0, 0)
    return img.shape[:2]


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
