import React from 'react';
import { Upload, Focus, Sparkles } from 'lucide-react';

export default function ProgressSteps({ currentStep }) {
  const steps = [
    { number: 1, label: 'Upload Images', icon: <Upload size={14} /> },
    { number: 2, label: 'Select Objects', icon: <Focus size={14} /> },
    { number: 3, label: 'View Result', icon: <Sparkles size={14} /> },
  ];

  return (
    <div className="stepper-container">
      {steps.map((step, index) => {
        const isActive = currentStep === step.number;
        const isCompleted = currentStep > step.number;
        
        return (
          <React.Fragment key={step.number}>
            {/* Step Node */}
            <div className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
              <div className="step-circle">
                {isCompleted ? '✓' : step.icon}
              </div>
              <span>{step.label}</span>
            </div>

            {/* Connecting line */}
            {index < steps.length - 1 && (
              <div className={`step-divider ${isCompleted ? 'filled' : ''}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
