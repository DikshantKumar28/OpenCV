import axios from 'axios';

// The client targets endpoints relative to the dev server since Vite is set up with a proxy.
// Fallback to localhost:8000 if needed, but relative works best through proxy.
const API_BASE = '';

/**
 * Upload target and optional reference images to run object detection.
 * @param {File} targetFile - The main image to remove objects from.
 * @param {File[]} referenceFiles - Array of reference images for background reconstruction.
 * @returns {Promise<Object>} API response with session_id, detections list, preview url, etc.
 */
export async function runDetection(targetFile, referenceFiles = [], conf = 0.15) {
  const formData = new FormData();
  
  // The first image in the list is always the target image
  formData.append('images', targetFile);
  
  // Append reference images
  referenceFiles.forEach((file) => {
    formData.append('images', file);
  });
  
  formData.append('conf', conf);

  const response = await axios.post(`${API_BASE}/api/detect`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Remove selected objects from a single image using OpenCV inpainting.
 * @param {string} sessionId - The session identifier.
 * @param {number[]} selectedIds - List of detection IDs to remove.
 * @returns {Promise<Object>} API response with output image name and url.
 */
export async function removeSingle(sessionId, selectedIds, params = {}) {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('selected_ids', JSON.stringify(selectedIds));
  
  if (params.maskDilation !== undefined) formData.append('mask_dilate_iterations', params.maskDilation);
  if (params.maskDilateKernel !== undefined) formData.append('mask_dilate_kernel', params.maskDilateKernel);
  if (params.inpaintRadius !== undefined) formData.append('inpaint_radius', params.inpaintRadius);
  if (params.useStrongInpaint !== undefined) formData.append('use_strong_inpaint', params.useStrongInpaint);
  if (params.removalMode !== undefined) formData.append('removal_mode', params.removalMode);

  const response = await axios.post(`${API_BASE}/api/remove-single`, formData);
  return response.data;
}

/**
 * Remove selected objects using multi-image background reconstruction.
 * @param {string} sessionId - The session identifier.
 * @param {number[]} selectedIds - List of detection IDs to remove.
 * @returns {Promise<Object>} API response with output image name and url.
 */
export async function removeMultiple(sessionId, selectedIds, params = {}) {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('selected_ids', JSON.stringify(selectedIds));
  
  if (params.maskDilation !== undefined) formData.append('mask_dilate_iterations', params.maskDilation);
  if (params.maskDilateKernel !== undefined) formData.append('mask_dilate_kernel', params.maskDilateKernel);
  if (params.inpaintRadius !== undefined) formData.append('inpaint_radius', params.inpaintRadius);
  if (params.useStrongInpaint !== undefined) formData.append('use_strong_inpaint', params.useStrongInpaint);
  if (params.removalMode !== undefined) formData.append('removal_mode', params.removalMode);

  const response = await axios.post(`${API_BASE}/api/remove-multiple`, formData);
  return response.data;
}

/**
 * Formats a relative output or mask URL into a full-path URL.
 * @param {string} relativePath - Path returned from the API (e.g. /output/xyz.jpg).
 * @returns {string} Fully qualified URL.
 */
export function getAssetUrl(relativePath) {
  if (!relativePath) return '';
  if (
    relativePath.startsWith('http://') ||
    relativePath.startsWith('https://') ||
    relativePath.startsWith('blob:') ||
    relativePath.startsWith('data:') ||
    relativePath.startsWith('file:')
  ) {
    return relativePath;
  }
  // Ensure correct slash prefixing for backend asset paths
  const path = relativePath.startsWith('/') ? relativePath : `/${relativePath}`;
  return `${API_BASE}${path}`;
}
