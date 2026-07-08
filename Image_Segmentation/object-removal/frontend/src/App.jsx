import React, { useState } from 'react';
import ProgressSteps from './components/ProgressSteps';
import UploadZone from './components/UploadZone';
import ImageCanvas from './components/ImageCanvas';
import ObjectPanel from './components/ObjectPanel';
import BeforeAfterSlider from './components/BeforeAfterSlider';
import ActionButtons from './components/ActionButtons';
import { runDetection, removeSingle, removeMultiple } from './api/client';
import { Sparkles, AlertTriangle, Cpu, Landmark, Sliders, Loader2 } from 'lucide-react';

export default function App() {
  // Navigation & Pipeline State
  const [currentStep, setCurrentStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Files State
  const [targetFile, setTargetFile] = useState(null);
  const [referenceFiles, setReferenceFiles] = useState([]);

  // Session & Detection State
  const [sessionId, setSessionId] = useState('');
  const [mode, setMode] = useState('single');
  const [detections, setDetections] = useState([]);
  const [previewName, setPreviewName] = useState('');
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);

  // Selection & Output State
  const [selectedIds, setSelectedIds] = useState([]);
  const [outputUrl, setOutputUrl] = useState('');
  const [methodUsed, setMethodUsed] = useState('');
  const [warningText, setWarningText] = useState('');
  const [loadingStep, setLoadingStep] = useState('');

  // Tuning Configuration
  const [detectionSensitivity, setDetectionSensitivity] = useState(0.15);
  const [maskDilation, setMaskDilation] = useState(4);
  const [maskDilateKernel, setMaskDilateKernel] = useState(9);
  const [inpaintRadius, setInpaintRadius] = useState(15);
  const [useStrongInpaint, setUseStrongInpaint] = useState(true);
  const [removalMode, setRemovalMode] = useState('auto');

  // ════════════════════════════════════════════════════════════════════════
  // Pipeline Handlers
  // ════════════════════════════════════════════════════════════════════════

  // Step 1 -> 2: Upload images and execute YOLO detection
  const handleUploadAndDetect = async () => {
    if (!targetFile) return;
    setIsProcessing(true);
    setErrorMsg('');
    setLoadingStep('Detecting objects');
    try {
      const data = await runDetection(targetFile, referenceFiles, detectionSensitivity);
      setSessionId(data.session_id);
      setMode(data.mode);
      setDetections(data.detections || []);
      setPreviewName(data.preview_name);
      setImageWidth(data.image_width);
      setImageHeight(data.image_height);
      
      if (data.detections && data.detections.length > 0) {
        setSelectedIds(data.detections.map(d => d.object_id));
      } else {
        setSelectedIds([]);
      }
      
      setCurrentStep(2);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'Failed to connect to backend server. Make sure FastAPI server is running on http://localhost:8000.');
    } finally {
      setIsProcessing(false);
      setLoadingStep('');
    }
  };

  // Step 2 -> 3 (Option A): OpenCV Inpainting
  const handleInpaint = async () => {
    if (!sessionId || selectedIds.length === 0) return;
    setIsProcessing(true);
    setErrorMsg('');
    setLoadingStep('Preparing mask');
    setMethodUsed('');
    setWarningText('');
    try {
      const params = {
        maskDilation,
        maskDilateKernel,
        inpaintRadius,
        useStrongInpaint,
        removalMode,
      };
      const data = await removeSingle(sessionId, selectedIds, params);
      setOutputUrl(data.output_url);
      setMethodUsed(data.method_used || '');
      setWarningText(data.warning || '');
      setCurrentStep(3);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'Single-image inpainting failed.');
    } finally {
      setIsProcessing(false);
      setLoadingStep('Finalizing output');
      setTimeout(() => setLoadingStep(''), 600);
    }
  };

  // Step 2 -> 3 (Option B): Multi-Image Background Reconstruction
  const handleReconstruct = async () => {
    if (!sessionId || selectedIds.length === 0) return;
    setIsProcessing(true);
    setErrorMsg('');
    setLoadingStep('Aligning reference images');
    setMethodUsed('');
    setWarningText('');
    try {
      const params = {
        maskDilation,
        maskDilateKernel,
        inpaintRadius,
        useStrongInpaint,
        removalMode,
      };
      const data = await removeMultiple(sessionId, selectedIds, params);
      setOutputUrl(data.output_url);
      setMethodUsed(data.method_used || '');
      setWarningText(data.warning || '');
      setCurrentStep(3);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'Multi-image background reconstruction failed.');
    } finally {
      setIsProcessing(false);
      setLoadingStep('Finalizing output');
      setTimeout(() => setLoadingStep(''), 600);
    }
  };

  // Selection helpers
  const handleToggleSelection = (id) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    setSelectedIds(detections.map((d) => d.object_id));
  };

  const handleDeselectAll = () => {
    setSelectedIds([]);
  };

  // Reset complete pipeline state
  const handleReset = () => {
    setCurrentStep(1);
    setIsProcessing(false);
    setErrorMsg('');
    setTargetFile(null);
    setReferenceFiles([]);
    setSessionId('');
    setMode('single');
    setDetections([]);
    setPreviewName('');
    setSelectedIds([]);
    setOutputUrl('');
  };

  // Return to selection step to try different settings
  const handleBackToSelection = () => {
    setCurrentStep(2);
    setOutputUrl('');
    setMethodUsed('');
    setWarningText('');
  };

  return (
    <div className="app-container">
      {/* Premium Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">✨</div>
          <span>Magic Eraser</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.12)', border: '1px solid rgba(6, 182, 212, 0.2)', color: '#22d3ee' }}>
            YOLOv11 + SAM2
          </span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>v1.0.0</span>
        </div>
      </header>

      {/* Loading Overlay */}
      {isProcessing && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(9, 9, 11, 0.7)',
          backdropFilter: 'blur(8px)',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff'
        }}>
          <Loader2 size={48} className="spinner" style={{ animation: 'spin 2s linear infinite', color: 'var(--accent-primary)', marginBottom: '16px' }} />
          <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '8px' }}>Processing...</h3>
          <p style={{ color: 'var(--text-secondary)' }}>{loadingStep || 'Please wait'}</p>
        </div>
      )}

      {/* Main Container */}
      <main className="main-content">
        {/* Left Side: Canvas / Upload Workspace */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <ProgressSteps currentStep={currentStep} />

          {/* Error Alert Box */}
          {errorMsg && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '16px 20px',
                background: 'rgba(239, 68, 68, 0.08)',
                border: '1px solid rgba(239, 68, 68, 0.25)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--accent-danger)',
                fontSize: '0.9rem',
                lineHeight: '1.4'
              }}
            >
              <AlertTriangle size={20} style={{ flexShrink: 0 }} />
              <div>
                <strong style={{ display: 'block', marginBottom: '2px' }}>Pipeline Error</strong>
                {errorMsg}
              </div>
            </div>
          )}

          {/* Dynamic Stage Render */}
          {currentStep === 1 && (
            <div className="glass-panel">
              <div className="panel-header">
                <h3 className="panel-title">Workspace Uploads</h3>
              </div>
              <div className="panel-body">
                <UploadZone
                  targetFile={targetFile}
                  setTargetFile={setTargetFile}
                  referenceFiles={referenceFiles}
                  setReferenceFiles={setReferenceFiles}
                  onNext={handleUploadAndDetect}
                  disabled={isProcessing}
                />
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <ImageCanvas
              targetUrl={previewName ? `/output/${previewName}` : URL.createObjectURL(targetFile)}
              detections={detections}
              selectedIds={selectedIds}
              onToggleSelection={handleToggleSelection}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
              isProcessing={isProcessing}
            />
          )}

          {currentStep === 3 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <BeforeAfterSlider
                beforeUrl={previewName ? `/output/${previewName}` : URL.createObjectURL(targetFile)}
                afterUrl={outputUrl}
              />
              <div className="glass-panel" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '8px' }}>Result Metadata</h4>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
                        Useful details from the last removal operation.
                      </p>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {methodUsed && (
                        <span style={{ padding: '8px 12px', borderRadius: '999px', background: 'rgba(99, 102, 241, 0.12)', color: 'var(--accent-primary)', fontWeight: 600, fontSize: '0.82rem' }}>
                          Method: {methodUsed}
                        </span>
                      )}
                      {warningText && (
                        <span style={{ padding: '8px 12px', borderRadius: '999px', background: 'rgba(239, 68, 68, 0.12)', color: 'var(--accent-danger)', fontWeight: 600, fontSize: '0.82rem' }}>
                          Warning: {warningText}
                        </span>
                      )}
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                    <div style={{ padding: '14px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <strong style={{ display: 'block', marginBottom: '6px', color: 'var(--text-primary)' }}>Removal Mode</strong>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{removalMode.replace('_', ' ').toUpperCase()}</span>
                    </div>
                    <div style={{ padding: '14px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <strong style={{ display: 'block', marginBottom: '6px', color: 'var(--text-primary)' }}>Mask Dilation</strong>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{maskDilation} iterations</span>
                    </div>
                    <div style={{ padding: '14px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <strong style={{ display: 'block', marginBottom: '6px', color: 'var(--text-primary)' }}>Inpaint Radius</strong>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{inpaintRadius}px</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="glass-panel" style={{ padding: '20px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '12px' }}>Debug Outputs</h4>
                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Mask Used</span>
                    <img src={`/output/debug_mask.png?t=${new Date().getTime()}`} alt="Debug Mask" style={{ maxHeight: '120px', borderRadius: '4px', marginTop: '8px' }} onError={(e) => e.target.style.display='none'} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Telea Result</span>
                    <img src={`/output/debug_telea.png?t=${new Date().getTime()}`} alt="Debug Telea" style={{ maxHeight: '120px', borderRadius: '4px', marginTop: '8px' }} onError={(e) => e.target.style.display='none'} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>NS Result</span>
                    <img src={`/output/debug_ns.png?t=${new Date().getTime()}`} alt="Debug NS" style={{ maxHeight: '120px', borderRadius: '4px', marginTop: '8px' }} onError={(e) => e.target.style.display='none'} />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Operations Side Panel */}
        <div className="sidebar-panel">
          {currentStep === 1 ? (
            /* Informational / Instructional Panel for Step 1 */
            <>
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={18} style={{ color: 'var(--accent-primary)' }} />
                AI Tech Stack
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '20px' }}>
                Magic Eraser combines cutting-edge vision layers to intelligently fill or replace obscured portions of images.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-primary)', marginTop: '6px', flexShrink: 0 }}></div>
                  <div>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: '600', marginBottom: '4px' }}>YOLO11 Segmentation</h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      Locates bounding boxes and segments up to 80 different object classes instantly.
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-secondary)', marginTop: '6px', flexShrink: 0 }}></div>
                  <div>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: '600', marginBottom: '4px' }}>Segment Anything (SAM)</h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      Optionally refines coarse boundaries down to precise pixel-accurate contours.
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-success)', marginTop: '6px', flexShrink: 0 }}></div>
                  <div>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: '600', marginBottom: '4px' }}>Homography Alignment</h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      Aligns feature points matching perspective changes across reference pictures.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sliders size={16} style={{ color: 'var(--accent-secondary)' }} />
                Advanced Settings (Detection)
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>Sensitivity:</span>
                  <span>{detectionSensitivity}</span>
                </label>
                <input
                  type="range"
                  min="0.05" max="0.50" step="0.05"
                  value={detectionSensitivity}
                  onChange={(e) => setDetectionSensitivity(parseFloat(e.target.value))}
                  disabled={isProcessing}
                />
              </div>
            </div>
          </>
          ) : (
            /* Active Controls for Steps 2 & 3 */
            <>
              <ObjectPanel
                detections={detections}
                selectedIds={selectedIds}
                onToggleSelection={handleToggleSelection}
                onSelectAll={handleSelectAll}
                onDeselectAll={handleDeselectAll}
              />
              
              <div className="glass-panel" style={{ padding: '20px 24px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sliders size={16} style={{ color: 'var(--accent-secondary)' }} />
                  Advanced Settings (Removal)
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Mask Dilation Iterations:</span>
                      <span>{maskDilation}</span>
                    </label>
                    <input type="range" min="1" max="10" step="1" value={maskDilation} onChange={(e) => setMaskDilation(parseInt(e.target.value))} disabled={isProcessing} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Mask Dilation Kernel:</span>
                      <span>{maskDilateKernel}</span>
                    </label>
                    <input type="range" min="3" max="21" step="2" value={maskDilateKernel} onChange={(e) => setMaskDilateKernel(parseInt(e.target.value))} disabled={isProcessing} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Inpaint Radius:</span>
                      <span>{inpaintRadius}</span>
                    </label>
                    <input type="range" min="3" max="31" step="2" value={inpaintRadius} onChange={(e) => setInpaintRadius(parseInt(e.target.value))} disabled={isProcessing} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Removal Mode:
                    </label>
                    <select
                      value={removalMode}
                      onChange={(e) => setRemovalMode(e.target.value)}
                      disabled={isProcessing}
                      className="control-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="opencv_fast">OpenCV Fast</option>
                      <option value="opencv_strong">OpenCV Strong</option>
                      <option value="reference_replacement">Reference Replacement</option>
                      <option value="ai_inpainting">AI Inpainting</option>
                    </select>
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    <input type="checkbox" checked={useStrongInpaint} onChange={(e) => setUseStrongInpaint(e.target.checked)} disabled={isProcessing} />
                    Use Strong Inpaint (Telea + NS)
                  </label>
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '20px 24px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  Removal Actions
                </h4>
                <ActionButtons
                  mode={mode}
                  selectedIds={selectedIds}
                  onInpaint={handleInpaint}
                  onReconstruct={handleReconstruct}
                  onReset={handleReset}
                  onBackToSelection={handleBackToSelection}
                  outputUrl={outputUrl}
                  isProcessing={isProcessing}
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
