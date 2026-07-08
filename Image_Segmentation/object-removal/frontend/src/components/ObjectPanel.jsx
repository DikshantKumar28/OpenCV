import React, { useState } from 'react';
import { Layers, CheckSquare, Square, Check, Info, Zap, Wand2, Settings } from 'lucide-react';

export default function ObjectPanel({
  detections,
  selectedIds,
  onToggleSelection,
  onSelectAll,
  onDeselectAll
}) {
  const hasDetections = detections && detections.length > 0;
  const [removalMethod, setRemovalMethod] = useState('auto');
  const [confidenceFilter, setConfidenceFilter] = useState(0.15);

  return (
    <div className="glass-panel object-panel">
      <div className="panel-header">
        <h3 className="panel-title">
          <Layers size={18} />
          Detected Objects
        </h3>
        {hasDetections && (
          <span className="badge" style={{ fontSize: '0.7rem' }}>
            {detections.length} Found
          </span>
        )}
      </div>

      <div className="panel-body object-panel-body">
        {hasDetections ? (
          <>
            {/* Quick Actions (Select/Deselect All) */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ flex: 1, padding: '10px 12px', fontSize: '0.8rem' }}
                onClick={onSelectAll}
              >
                Select All
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ flex: 1, padding: '10px 12px', fontSize: '0.8rem' }}
                onClick={onDeselectAll}
              >
                Deselect All
              </button>
            </div>

            {/* Removal Settings Section */}
            <div className="object-removal-settings">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', fontWeight: '600', color: 'var(--text-secondary)' }}>
                <Settings size={14} /> Removal Settings
              </div>
              
              {/* Removal Method */}
              <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Method</label>
                <select 
                  value={removalMethod} 
                  onChange={(e) => setRemovalMethod(e.target.value)}
                  className="control-select control-select-compact"
                >
                  <option value="auto">Auto (Telea/NS)</option>
                  <option value="telea">Telea</option>
                  <option value="ns">Navier-Stokes</option>
                </select>
              </div>

              {/* Confidence Threshold */}
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Confidence: {Math.round(confidenceFilter * 100)}%
                </label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.05"
                  value={confidenceFilter}
                  onChange={(e) => setConfidenceFilter(parseFloat(e.target.value))}
                  style={{
                    width: '100%',
                    cursor: 'pointer',
                    accentColor: 'var(--accent-primary)'
                  }}
                />
              </div>
            </div>

            {/* Checklist of Detections */}
            <div className="detection-list">
              {detections.map((det) => {
                const isSelected = selectedIds.includes(det.object_id);
                return (
                  <div
                    key={det.object_id}
                    className={`detection-item ${isSelected ? 'selected' : ''}`}
                    onClick={() => onToggleSelection(det.object_id)}
                  >
                    <div className="detection-label-group">
                      <div className="detection-checkbox">
                        {isSelected && <Check size={14} strokeWidth={3} />}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span className="class-label" style={{ fontSize: '0.95rem' }}>
                          {det.class_name} <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 'normal' }}>#{det.object_id + 1}</span>
                        </span>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Conf: {Math.round(det.confidence * 100)}%</span>
                      </div>
                    </div>
                    <span className="conf-pill" style={{ padding: '4px 10px' }}>
                      {isSelected ? <Zap size={14} /> : <Square size={14} strokeWidth={2} />}
                    </span>
                  </div>
                );
              })}
            </div>
            
            <div className="help-text" style={{ marginTop: '20px' }}>
              <Info size={14} />
              <span>
                Select objects from the list or click directly on the image boxes to mark them for removal.
              </span>
            </div>
          </>
        ) : (
          <div className="empty-detections">
            <Info size={32} style={{ color: 'var(--text-muted)', marginBottom: '8px' }} />
            <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>No Objects Detected</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              YOLO could not find any segmentable objects with confidence &gt; 25%. You can upload a different image.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
