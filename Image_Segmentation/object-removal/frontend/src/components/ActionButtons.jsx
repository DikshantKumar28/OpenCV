import React from 'react';
import { Eraser, Sparkles, RefreshCw, Download, ArrowLeft } from 'lucide-react';
import { getAssetUrl } from '../api/client';

export default function ActionButtons({
  mode,
  selectedIds,
  onInpaint,
  onReconstruct,
  onReset,
  onBackToSelection,
  outputUrl,
  isProcessing
}) {
  const hasSelection = selectedIds.length > 0;
  const isMultiMode = mode === 'multiple';

  return (
    <div className="action-box">
      {/* If we have processed result, show Download and Reset */}
      {outputUrl ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <a
            href={getAssetUrl(outputUrl)}
            download="magic-eraser-result.png"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
            style={{ width: '100%', textDecoration: 'none' }}
          >
            <Download size={18} /> Download Result
          </a>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onBackToSelection}
            style={{ width: '100%' }}
          >
            <ArrowLeft size={16} /> Try Different Settings
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onReset}
            style={{ width: '100%' }}
          >
            <RefreshCw size={16} /> Process Another Image
          </button>
        </div>
      ) : (
        /* Otherwise, show standard processing actions */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* Multi-Image Reconstruction (Highlight if in Multi Mode) */}
          <button
            type="button"
            className={`btn ${isMultiMode && hasSelection ? 'btn-accent' : 'btn-disabled'}`}
            onClick={onReconstruct}
            disabled={!isMultiMode || !hasSelection || isProcessing}
            style={{ width: '100%' }}
          >
            <Sparkles size={18} />
            Reconstruct Background
          </button>
          
          {/* Single-Image Inpainting */}
          <button
            type="button"
            className={`btn ${(!isMultiMode || !isMultiMode) && hasSelection ? 'btn-primary' : 'btn-secondary'} ${!hasSelection ? 'btn-disabled' : ''}`}
            onClick={onInpaint}
            disabled={!hasSelection || isProcessing}
            style={{ width: '100%' }}
          >
            <Eraser size={18} />
            Inpaint Selected Objects
          </button>

          {!isMultiMode && (
            <div className="help-text" style={{ justifyContent: 'center' }}>
              💡 Want perfect background replication? Upload reference photos in step 1.
            </div>
          )}

          {isMultiMode && !hasSelection && (
            <div className="help-text" style={{ justifyContent: 'center', color: 'var(--accent-warning)' }}>
              ⚠️ Select one or more objects to enable removal.
            </div>
          )}

          <div style={{ display: 'flex', gap: '10px', marginTop: '8px' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onReset}
              style={{ flex: 1 }}
              disabled={isProcessing}
            >
              <ArrowLeft size={16} /> Back
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
