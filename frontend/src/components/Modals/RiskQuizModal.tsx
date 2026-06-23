import React, { useState } from 'react';

interface RiskQuizModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (level: 'Low' | 'Medium' | 'High', score: number) => void;
}

export const RiskQuizModal: React.FC<RiskQuizModalProps> = ({ isOpen, onClose, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [q1, setQ1] = useState<number | null>(null);
  const [q2, setQ2] = useState<number | null>(null);
  const [q3, setQ3] = useState<number | null>(null);
  const [sliderVal, setSliderVal] = useState(50);
  const [sectors, setSectors] = useState<string[]>([]);

  if (!isOpen) return null;

  const stepsCount = 5;

  const toggleSector = (sector: string) => {
    if (sectors.includes(sector)) {
      setSectors(sectors.filter((s) => s !== sector));
    } else {
      setSectors([...sectors, sector]);
    }
  };

  const handleNext = () => {
    if (currentStep < stepsCount - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // compute scores according to vanilla formulas
    // Weight = 1 for all steps (weightSum = 5)
    // idx 0, 1, 2 scores: q1, q2, q3 (default to 0 if not selected)
    // idx 3 (slider): (sliderVal / 100) * 3
    // idx 4 (sectors): selected options count? Wait, in vanilla code,
    // sectors has class="quiz-option" and each option has data-score="1".
    // If selected, score is added.
    const s1 = q1 ?? 0;
    const s2 = q2 ?? 0;
    const s3 = q3 ?? 0;
    const s4 = (sliderVal / 100) * 3;
    const s5 = sectors.length > 0 ? 1 : 0; // Each sector is scored as 1 in vanilla if selected (actually options selected have score data-score=1).

    const scoreSum = s1 + s2 + s3 + s4 + s5;
    const weightSum = 5;

    const normalized = Math.round((scoreSum / (3 * weightSum)) * 100);
    const level = normalized < 35 ? 'Low' : normalized < 70 ? 'Medium' : 'High';

    onComplete(level, normalized);
    onClose();
  };

  return (
    <div className="modal" style={{ display: 'flex' }} onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div className="modal-content">
        <div className="modal-header">
          <h3>Risk Assessment Quiz</h3>
          <span className="close" onClick={onClose}>&times;</span>
        </div>
        <div className="modal-body">
          <div className="quiz-progress">
            <div 
              className="quiz-progress-bar" 
              style={{ width: `${(currentStep / (stepsCount - 1)) * 100}%` }}
            />
          </div>
          <form id="risk-quiz-form" onSubmit={handleSubmit}>
            <div className="quiz-steps" id="quiz-steps">
              
              {/* Step 1 */}
              {currentStep === 0 && (
                <div className="quiz-step active">
                  <div className="quiz-q">How do you feel about market ups and downs?</div>
                  <div className="quiz-options">
                    <div 
                      className={`quiz-option ${q1 === 0 ? 'selected' : ''}`} 
                      onClick={() => setQ1(0)}
                    >
                      <span className="quiz-emoji">😬</span> I dislike volatility
                    </div>
                    <div 
                      className={`quiz-option ${q1 === 1 ? 'selected' : ''}`} 
                      onClick={() => setQ1(1)}
                    >
                      <span className="quiz-emoji">🙂</span> I can tolerate some
                    </div>
                    <div 
                      className={`quiz-option ${q1 === 2 ? 'selected' : ''}`} 
                      onClick={() => setQ1(2)}
                    >
                      <span className="quiz-emoji">😎</span> Comfortable with swings
                    </div>
                    <div 
                      className={`quiz-option ${q1 === 3 ? 'selected' : ''}`} 
                      onClick={() => setQ1(3)}
                    >
                      <span className="quiz-emoji">🚀</span> Love the action
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2 */}
              {currentStep === 1 && (
                <div className="quiz-step active">
                  <div className="quiz-q">Your investment horizon</div>
                  <div className="quiz-options">
                    <div 
                      className={`quiz-option ${q2 === 0 ? 'selected' : ''}`} 
                      onClick={() => setQ2(0)}
                    >
                      Less than 2 years
                    </div>
                    <div 
                      className={`quiz-option ${q2 === 1 ? 'selected' : ''}`} 
                      onClick={() => setQ2(1)}
                    >
                      2-5 years
                    </div>
                    <div 
                      className={`quiz-option ${q2 === 2 ? 'selected' : ''}`} 
                      onClick={() => setQ2(2)}
                    >
                      5-10 years
                    </div>
                    <div 
                      className={`quiz-option ${q2 === 3 ? 'selected' : ''}`} 
                      onClick={() => setQ2(3)}
                    >
                      More than 10 years
                    </div>
                  </div>
                </div>
              )}

              {/* Step 3 */}
              {currentStep === 2 && (
                <div className="quiz-step active">
                  <div className="quiz-q">If your portfolio fell 20% in a month, you would…</div>
                  <div className="quiz-options">
                    <div 
                      className={`quiz-option ${q3 === 0 ? 'selected' : ''}`} 
                      onClick={() => setQ3(0)}
                    >
                      Sell everything
                    </div>
                    <div 
                      className={`quiz-option ${q3 === 1 ? 'selected' : ''}`} 
                      onClick={() => setQ3(1)}
                    >
                      Sell some
                    </div>
                    <div 
                      className={`quiz-option ${q3 === 2 ? 'selected' : ''}`} 
                      onClick={() => setQ3(2)}
                    >
                      Do nothing
                    </div>
                    <div 
                      className={`quiz-option ${q3 === 3 ? 'selected' : ''}`} 
                      onClick={() => setQ3(3)}
                    >
                      Buy more
                    </div>
                  </div>
                </div>
              )}

              {/* Step 4 */}
              {currentStep === 3 && (
                <div className="quiz-step active">
                  <div className="quiz-q">Allocate risk using the slider</div>
                  <div className="quiz-range">
                    <input 
                      type="range" 
                      min="0" 
                      max="100" 
                      value={sliderVal} 
                      onChange={(e) => setSliderVal(parseInt(e.target.value))}
                      id="quiz-risk-slider" 
                    />
                    <div className="quiz-hint">0 = Safety first • 100 = Max growth</div>
                  </div>
                </div>
              )}

              {/* Step 5 */}
              {currentStep === 4 && (
                <div className="quiz-step active">
                  <div className="quiz-q">Pick sectors you enjoy following</div>
                  <div className="quiz-options">
                    <div 
                      className={`quiz-option ${sectors.includes('Tech') ? 'selected' : ''}`} 
                      onClick={() => toggleSector('Tech')}
                    >
                      Tech
                    </div>
                    <div 
                      className={`quiz-option ${sectors.includes('Healthcare') ? 'selected' : ''}`} 
                      onClick={() => toggleSector('Healthcare')}
                    >
                      Healthcare
                    </div>
                    <div 
                      className={`quiz-option ${sectors.includes('Energy') ? 'selected' : ''}`} 
                      onClick={() => toggleSector('Energy')}
                    >
                      Energy
                    </div>
                    <div 
                      className={`quiz-option ${sectors.includes('Real Estate') ? 'selected' : ''}`} 
                      onClick={() => toggleSector('Real Estate')}
                    >
                      Real Estate
                    </div>
                  </div>
                  <div className="quiz-hint">This won't change your risk score much, but helps tailor ideas.</div>
                </div>
              )}

            </div>
            <div className="quiz-nav" style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button 
                type="button" 
                className="btn btn-secondary" 
                onClick={handlePrev} 
                disabled={currentStep === 0}
              >
                Back
              </button>
              <button 
                type="button" 
                className="btn btn-primary" 
                onClick={handleNext}
                disabled={currentStep === stepsCount - 1}
                style={{ display: currentStep === stepsCount - 1 ? 'none' : 'block' }}
              >
                Next
              </button>
              {currentStep === stepsCount - 1 && (
                <button type="submit" className="btn btn-primary">
                  Save Profile
                </button>
              )}
            </div>
            <div className="modal-actions">
              <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
