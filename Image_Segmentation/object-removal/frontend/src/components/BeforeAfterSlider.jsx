import React, { useState, useRef, useEffect } from 'react';
import { getAssetUrl } from '../api/client';

export default function BeforeAfterSlider({ beforeUrl, afterUrl }) {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage (0-100)
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  const handleMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    let position = (x / rect.width) * 100;
    if (position < 0) position = 0;
    if (position > 100) position = 100;
    setSliderPosition(position);
  };

  const handleMouseMove = (e) => {
    handleMove(e.clientX);
  };

  const handleTouchMove = (e) => {
    if (e.touches && e.touches[0]) {
      handleMove(e.touches[0].clientX);
    }
  };

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      isDragging.current = false;
    };
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  return (
    <div
      className="slider-container"
      ref={containerRef}
      onMouseMove={(e) => isDragging.current && handleMouseMove(e)}
      onTouchMove={handleTouchMove}
      onMouseDown={handleMouseDown}
      style={{ cursor: 'ew-resize' }}
    >
      {/* Before Image (Left side / Background) */}
      <img
        src={getAssetUrl(beforeUrl)}
        alt="Before"
        className="slider-img slider-before"
        draggable={false}
      />
      <span className="slider-label before">Before</span>

      {/* After Image (Right side / Foreground clipped dynamically) */}
      <img
        src={getAssetUrl(afterUrl)}
        alt="After"
        className="slider-img slider-after"
        style={{
          clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)`
        }}
        draggable={false}
      />
      <span className="slider-label after">After</span>

      {/* Vertical Slider Bar */}
      <div
        className="slider-handle"
        style={{ left: `${sliderPosition}%` }}
      >
        <div className="slider-button">↔</div>
      </div>
    </div>
  );
}
