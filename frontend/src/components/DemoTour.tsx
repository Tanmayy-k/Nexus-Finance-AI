import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDemoStore } from '../store/useDemoStore';
import { useNavigate } from 'react-router-dom';

const TOUR_STEPS = [
  {
    id: 'step-1',
    title: 'Financial Health Score',
    content: 'Powered by Random Forest models, this score gives you a real-time snapshot of your financial well-being.',
    targetId: 'score-target', // We need to add this ID to DashboardPage
    section: 'analytics'
  },
  {
    id: 'step-2',
    title: 'Explainable AI Insights',
    content: 'Using SHAP values, Nexus transparently explains exactly which factors are impacting your score.',
    targetId: 'shap-target',
    section: 'analytics'
  },
  {
    id: 'step-3',
    title: 'Nexus AI Copilot',
    content: 'Your 24/7 personal financial advisor. Ask questions about your spending, get tailored advice, and more.',
    targetId: 'floating-ai-toggle',
    section: 'dashboard'
  },
  {
    id: 'step-4',
    title: 'Predictive Forecasting',
    content: 'Our models analyze your transaction history to accurately predict your future spending and cash flow.',
    targetId: 'forecast-target',
    section: 'analytics'
  },
  {
    id: 'step-5',
    title: 'Smart Budget Envelopes',
    content: 'Real-world zero-based budgeting. See exactly how much you have left before you overspend.',
    targetId: 'nav-budget',
    section: 'budget'
  },
  {
    id: 'step-6',
    title: 'Investment Intelligence',
    content: 'Get personalized portfolio recommendations based on your risk profile and market conditions.',
    targetId: 'nav-investments',
    section: 'investments'
  }
];

export const DemoTour: React.FC<{ setActiveSection: (s: string) => void }> = ({ setActiveSection }) => {
  const { isTourActive, currentTourStep, endTour, nextTourStep, prevTourStep } = useDemoStore();
  const navigate = useNavigate();
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);

  const step = useMemo(() => TOUR_STEPS[currentTourStep], [currentTourStep]);

  useEffect(() => {
    if (!isTourActive) return;

    // Switch section automatically based on tour step
    if (step && step.section) {
      setActiveSection(step.section);
    }

    const updateRect = () => {
      if (!step) return;
      const el = document.getElementById(step.targetId);
      if (el) {
        setTargetRect(el.getBoundingClientRect());
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        setTargetRect(null);
      }
    };

    // Small delay to allow React to render the new section
    const timer = setTimeout(updateRect, 300);
    window.addEventListener('resize', updateRect);
    window.addEventListener('scroll', updateRect, true); // Use capture phase to catch scroll on any child container
    
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', updateRect);
      window.removeEventListener('scroll', updateRect, true);
    };
  }, [isTourActive, step, setActiveSection]);

  if (!isTourActive || !step) return null;

  const isLast = currentTourStep === TOUR_STEPS.length - 1;

  const handleNext = () => {
    if (isLast) {
      useDemoStore.getState().exitDemo();
      navigate('/login#signup');
    } else {
      nextTourStep();
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: 'fixed',
          inset: 0,
          width: '100%',
          height: '100%',
          zIndex: 9999,
          pointerEvents: 'none', // Allow clicks to pass through if needed, but we intercept below
        }}
      >
        {/* Dark overlay with true spotlight cutout using 4 surrounding divs */}
        {targetRect ? (
          <>
            {/* Top */}
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: Math.max(0, targetRect.top - 12), backgroundColor: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', pointerEvents: 'auto' }} />
            {/* Bottom */}
            <div style={{ position: 'absolute', top: targetRect.bottom + 12, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', pointerEvents: 'auto' }} />
            {/* Left */}
            <div style={{ position: 'absolute', top: targetRect.top - 12, left: 0, width: Math.max(0, targetRect.left - 12), height: targetRect.height + 24, backgroundColor: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', pointerEvents: 'auto' }} />
            {/* Right */}
            <div style={{ position: 'absolute', top: targetRect.top - 12, left: targetRect.right + 12, right: 0, height: targetRect.height + 24, backgroundColor: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', pointerEvents: 'auto' }} />
            {/* Invisible center blocker to prevent clicking underlying elements */}
            <div style={{ position: 'absolute', top: targetRect.top - 12, left: targetRect.left - 12, width: targetRect.width + 24, height: targetRect.height + 24, pointerEvents: 'auto' }} />
          </>
        ) : (
          <div style={{
            position: 'absolute',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.7)',
            backdropFilter: 'blur(4px)',
            pointerEvents: 'auto',
          }} />
        )}

        {/* Highlight target clone or outline (optional, for now we just show the card pointing to it) */}
        {targetRect && (
          <motion.div
            layout
            initial={false}
            animate={{
              x: targetRect.left - 8,
              y: targetRect.top - 8,
              width: targetRect.width + 16,
              height: targetRect.height + 16,
            }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            style={{
              position: 'absolute',
              border: '2px solid #6366f1',
              borderRadius: '12px',
              boxShadow: '0 0 0 4px rgba(99, 102, 241, 0.2), 0 0 20px rgba(99, 102, 241, 0.4)',
              zIndex: 10000,
              backgroundColor: 'transparent',
              pointerEvents: 'none'
            }}
          />
        )}

        {/* Floating Instruction Card */}
        <motion.div
          layout
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ 
            opacity: 1, 
            scale: 1, 
            y: 0,
            x: targetRect ? Math.min(Math.max(20, targetRect.left + (targetRect.width / 2) - 175), window.innerWidth - 370) : window.innerWidth / 2 - 175,
            top: targetRect ? Math.max(20, Math.min(window.innerHeight - 220, targetRect.bottom + 20 > window.innerHeight - 200 ? targetRect.top - 220 : targetRect.bottom + 20)) : window.innerHeight / 2 - 100,
          }}
          transition={{ type: "spring", stiffness: 150, damping: 20 }}
          style={{
            position: 'absolute',
            width: '350px',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(16px)',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
            zIndex: 10001,
            pointerEvents: 'auto',
            border: '1px solid rgba(255,255,255,0.2)',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#6366f1', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Step {currentTourStep + 1} of {TOUR_STEPS.length}
            </span>
            <button onClick={endTour} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: '4px' }}>
              <span className="material-icons" style={{ fontSize: '18px' }}>close</span>
            </button>
          </div>

          <div>
            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0f172a', margin: '0 0 8px 0' }}>{step.title}</h3>
            <p style={{ fontSize: '14px', color: '#475569', lineHeight: 1.5, margin: 0 }}>{step.content}</p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
            {/* Progress dots */}
            <div style={{ display: 'flex', gap: '4px' }}>
              {TOUR_STEPS.map((_, i) => (
                <div key={i} style={{ 
                  width: i === currentTourStep ? '16px' : '6px', 
                  height: '6px', 
                  borderRadius: '3px', 
                  backgroundColor: i === currentTourStep ? '#6366f1' : '#cbd5e1',
                  transition: 'all 0.3s ease'
                }} />
              ))}
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                onClick={() => {
                  if (currentTourStep === 0) {
                    useDemoStore.getState().exitDemo();
                    navigate('/');
                  } else {
                    prevTourStep();
                  }
                }}
                style={{ padding: '8px 12px', fontSize: '13px', borderRadius: '8px', border: '1px solid #e2e8f0', background: 'white', color: '#475569', cursor: 'pointer', fontWeight: 500 }}
              >
                Back
              </button>
              <button 
                onClick={handleNext}
                style={{ padding: '8px 16px', fontSize: '13px', borderRadius: '8px', border: 'none', background: '#6366f1', color: 'white', cursor: 'pointer', fontWeight: 600, boxShadow: '0 4px 10px rgba(99, 102, 241, 0.3)' }}
              >
                {isLast ? 'Get Started' : 'Next'}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
