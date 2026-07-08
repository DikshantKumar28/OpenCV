import React, { useRef } from 'react';
import { Upload, Image as ImageIcon, Files, X, Plus } from 'lucide-react';

export default function UploadZone({
  targetFile,
  setTargetFile,
  referenceFiles,
  setReferenceFiles,
  onNext,
  disabled
}) {
  const targetInputRef = useRef(null);
  const referencesInputRef = useRef(null);

  const handleTargetChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setTargetFile(e.target.files[0]);
    }
  };

  const handleReferencesChange = (e) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      setReferenceFiles((prev) => [...prev, ...filesArray]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleTargetDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setTargetFile(e.dataTransfer.files[0]);
    }
  };

  const handleReferencesDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      const filesArray = Array.from(e.dataTransfer.files);
      setReferenceFiles((prev) => [...prev, ...filesArray]);
    }
  };

  const removeReference = (index) => {
    setReferenceFiles((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="dropzone-container">
      {/* Target Image Upload (Required) */}
      <div>
        <h3 style={{ marginBottom: '10px', fontSize: '1rem', fontWeight: '600' }}>
          1. Upload Target Image <span style={{ color: 'var(--accent-primary)' }}>*</span>
        </h3>
        <div
          className="dropzone"
          onDragOver={handleDragOver}
          onDrop={handleTargetDrop}
          onClick={() => !disabled && targetInputRef.current?.click()}
          style={{
            borderColor: targetFile ? 'rgba(99, 102, 241, 0.4)' : 'rgba(255, 255, 255, 0.15)',
            background: targetFile ? 'rgba(99, 102, 241, 0.02)' : 'rgba(255, 255, 255, 0.01)',
          }}
        >
          <input
            type="file"
            ref={targetInputRef}
            onChange={handleTargetChange}
            accept="image/*"
            className="file-input"
            disabled={disabled}
          />
          {targetFile ? (
            <div style={{ width: '100%', position: 'relative' }}>
              <img
                src={URL.createObjectURL(targetFile)}
                alt="Target"
                style={{
                  maxHeight: '220px',
                  borderRadius: 'var(--radius-md)',
                  display: 'block',
                  margin: '0 auto',
                  objectFit: 'contain',
                }}
              />
              <button
                type="button"
                className="preview-remove"
                style={{ top: '-10px', right: '10px' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setTargetFile(null);
                }}
              >
                <X size={14} />
              </button>
              <div style={{ marginTop: '12px', fontSize: '0.9rem', fontWeight: '500' }}>
                {targetFile.name} ({(targetFile.size / 1024 / 1024).toFixed(2)} MB)
              </div>
            </div>
          ) : (
            <>
              <div className="dropzone-icon">
                <Upload size={28} />
              </div>
              <div className="dropzone-title">Drag & Drop your target image here</div>
              <div className="dropzone-text">
                Or click to browse. This is the main image from which you want to remove objects.
              </div>
            </>
          )}
        </div>
      </div>

      {/* Reference Images Upload (Optional, Multi) */}
      <div style={{ marginTop: '10px' }}>
        <h3 style={{ marginBottom: '10px', fontSize: '1rem', fontWeight: '600' }}>
          2. Reference Images <span style={{ color: 'var(--text-muted)', fontWeight: 'normal', fontSize: '0.85rem' }}>(Optional)</span>
        </h3>
        <div
          className="dropzone"
          onDragOver={handleDragOver}
          onDrop={handleReferencesDrop}
          onClick={() => !disabled && referencesInputRef.current?.click()}
          style={{ padding: '24px 16px' }}
        >
          <input
            type="file"
            ref={referencesInputRef}
            onChange={handleReferencesChange}
            accept="image/*"
            multiple
            className="file-input"
            disabled={disabled}
          />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
            <div className="dropzone-icon" style={{ width: '48px', height: '48px', fontSize: '1.4rem' }}>
              <Plus size={20} />
            </div>
            <div className="dropzone-title" style={{ fontSize: '0.95rem' }}>Add reference images</div>
            <div className="dropzone-text" style={{ fontSize: '0.8rem' }}>
              Upload images taken from slightly different positions to reconstruct background behind objects.
            </div>
          </div>
        </div>

        {referenceFiles.length > 0 && (
          <div className="previews-grid">
            {referenceFiles.map((file, index) => (
              <div key={index} className="preview-thumb">
                <span className="preview-badge">Ref #{index + 1}</span>
                <img src={URL.createObjectURL(file)} alt={`Ref ${index + 1}`} />
                <button
                  type="button"
                  className="preview-remove"
                  onClick={() => removeReference(index)}
                  disabled={disabled}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {targetFile && (
        <button
          type="button"
          className={`btn ${disabled ? 'btn-disabled' : 'btn-primary'}`}
          onClick={onNext}
          style={{ marginTop: '20px', width: '100%', padding: '14px' }}
          disabled={disabled}
        >
          {disabled ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></span>
              Analyzing (First run downloads AI models)...
            </span>
          ) : (
            <>
              <Upload size={18} /> Analyze & Detect Objects
            </>
          )}
        </button>
      )}
    </div>
  );
}
