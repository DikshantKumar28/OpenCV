import React, { useState, useRef, useEffect } from 'react';
import { getAssetUrl } from '../api/client';
import { Sparkles } from 'lucide-react';

export default function ImageCanvas({
  targetUrl,
  detections,
  selectedIds,
  onToggleSelection,
  imageWidth,
  imageHeight,
  isProcessing
}) {
  const [hoveredId, setHoveredId] = useState(null);
  const containerRef = useRef(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });

  // Calculate dynamic scaling from natural image size to displayed size
  const handleImageLoad = (e) => {
    const img = e.target;
    const rect = img.getBoundingClientRect();
    const naturalWidth = img.naturalWidth || imageWidth || rect.width;
    const naturalHeight = img.naturalHeight || imageHeight || rect.height;
    setScale({
      x: rect.width / naturalWidth,
      y: rect.height / naturalHeight
    });
  };

  // Keep scale updated on window resize
  useEffect(() => {
    const handleResize = () => {
      const img = containerRef.current?.querySelector('.target-image');
      if (img) {
        const rect = img.getBoundingClientRect();
        const naturalWidth = img.naturalWidth || imageWidth || rect.width;
        const naturalHeight = img.naturalHeight || imageHeight || rect.height;
        setScale({
          x: rect.width / naturalWidth,
          y: rect.height / naturalHeight
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [imageWidth, imageHeight, detections]);

  return (
    <div className="canvas-wrapper">
      <div className="interactive-image-container" ref={containerRef}>
        {/* Main Target Image */}
        <img
          src={getAssetUrl(targetUrl)}
          alt="Target"
          className="target-image"
          onLoad={handleImageLoad}
          draggable={false}
        />

        {/* Mask and Bounding Box Overlays */}
        {detections.map((det) => {
          const [x1, y1, x2, y2] = det.bbox;
          const left = x1 * scale.x;
          const top = y1 * scale.y;
          const width = (x2 - x1) * scale.x;
          const height = (y2 - y1) * scale.y;
          const isSelected = selectedIds.includes(det.object_id);
          const isHovered = hoveredId === det.object_id;

          return (
            <div key={det.object_id}>
              {/* Mask Overlay (Base64 PNG with alpha) */}
              {(isSelected || isHovered) && det.mask_b64 && (
                <img
                  src={`data:image/png;base64,${det.mask_b64}`}
                  alt={`Mask ${det.object_id}`}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    opacity: isSelected ? 0.75 : 0.45,
                    transition: 'opacity var(--transition-fast)',
                    mixBlendMode: 'normal'
                  }}
                />
              )}

              {/* Clickable Hover Bounding Box */}
              <div
                onClick={() => onToggleSelection(det.object_id)}
                onMouseEnter={() => setHoveredId(det.object_id)}
                onMouseLeave={() => setHoveredId(null)}
                style={{
                  position: 'absolute',
                  left: `${left}px`,
                  top: `${top}px`,
                  width: `${width}px`,
                  height: `${height}px`,
                  border: isSelected
                    ? '2px solid var(--accent-danger)'
                    : isHovered
                    ? '2.5px solid var(--accent-primary)'
                    : '1px dashed rgba(255, 255, 255, 0.25)',
                  backgroundColor: isSelected
                    ? 'rgba(239, 68, 68, 0.05)'
                    : isHovered
                    ? 'rgba(99, 102, 241, 0.05)'
                    : 'transparent',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  zIndex: isHovered ? 30 : 20,
                  transition: 'all var(--transition-fast)'
                }}
              >
                {/* Tiny Badge showing class name */}
                {isHovered && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '-24px',
                      left: '0px',
                      backgroundColor: 'rgba(9, 11, 16, 0.95)',
                      border: '1px solid var(--accent-primary)',
                      color: 'white',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      whiteSpace: 'nowrap',
                      boxShadow: 'var(--shadow-sm)',
                      pointerEvents: 'none',
                      textTransform: 'capitalize'
                    }}
                  >
                    {det.class_name} ({Math.round(det.confidence * 100)}%)
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Processing/Inpainting Backdrop overlay */}
        {isProcessing && (
          <div className="processing-overlay">
            <div className="spinner"></div>
            <div className="processing-text" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={20} style={{ color: 'var(--accent-primary)', animation: 'spin 3s infinite linear' }} />
              Reconstructing Background...
            </div>
            <div className="processing-sub">
              Running homography alignment and seamless Poisson blending.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
