import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/useAuthStore';
import { useDemoStore } from '../store/useDemoStore';
import { motion } from 'framer-motion';
import '../styles/landing.css';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  useEffect(() => {
    // If already logged in, redirect straight to dashboard
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const fadeInUp = {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  return (
    <div className="landing-body">
      <header className="landing-header">
        <div className="container">
          <div className="header-content">
            <div className="logo" style={{ cursor: 'pointer' }} onClick={() => navigate('/')}>
              <div className="logo-icon">
                <span className="material-icons">auto_awesome</span>
              </div>
              <div className="logo-text">Nexus Finance AI</div>
            </div>
            
            <nav className="nav-menu">
              <a href="#features" className="nav-item">Features</a>
              <a href="#solutions" className="nav-item">Solutions</a>
              <a href="#pricing" className="nav-item">Pricing</a>
              <a href="#resources" className="nav-item">Resources</a>
            </nav>
            
            <div className="header-actions">
              <button className="btn btn-outline" onClick={() => navigate('/login')}>Sign In</button>
              <Link to="/login#signup" className="btn btn-primary">Get Started</Link>
            </div>
          </div>
        </div>
      </header>
      
      <section className="hero">
        <div className="container">
          <motion.div 
            className="hero-content"
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
          >
            <motion.div className="hero-text" variants={fadeInUp}>
              <span className="hero-badge">Next Generation Intelligence</span>
              <h1 className="hero-title">Understand Your Money Like Never Before</h1>
              <p className="hero-subtitle">Nexus Finance AI leverages advanced machine learning to give you unprecedented visibility into your financial health, predictive forecasting, and personalized investment strategies.</p>
              
              <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => navigate('/login#signup')}>Get Started Free</button>
                <button className="btn btn-outline" onClick={() => {
                  useDemoStore.getState().startDemo();
                  navigate('/dashboard');
                }}>Explore Demo</button>
              </div>

              <div className="hero-stats">
                <div className="stat-item">
                  <div className="stat-value">AI Health Score</div>
                  <div className="stat-label">Random Forest Powered</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">SHAP Explainable</div>
                  <div className="stat-label">Transparent AI</div>
                </div>
              </div>
            </motion.div>
            
            <motion.div className="hero-visual" variants={fadeInUp}>
              <div className="dashboard-preview">
                <div className="preview-header">
                  <div className="preview-title">Financial Overview</div>
                  <div className="preview-actions">
                    <span className="material-icons">more_vert</span>
                  </div>
                </div>
                
                <div className="preview-content">
                  <div className="preview-card">
                    <div className="preview-label">Net Worth</div>
                    <div className="preview-metric">₹24,87,590</div>
                  </div>
                  <div className="preview-card">
                    <div className="preview-label">Liquid Assets</div>
                    <div className="preview-metric">₹3,12,450</div>
                  </div>
                  <div className="preview-card">
                    <div className="preview-label">Portfolio Return</div>
                    <div className="preview-metric">+8.4%</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      <section id="features" className="stories-container">
        <div className="container">
          
          {/* Story 1: AI Health Score */}
          <motion.div 
            className="story-section"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
          >
            <motion.div className="story-content" variants={fadeInUp}>
              <div className="story-badge">
                <span className="material-icons">health_and_safety</span>
                Financial Health Score
              </div>
              <h2 className="story-title">Quantify your financial wellness instantly.</h2>
              <p className="story-description">Our proprietary Random Forest model evaluates your income, spending, and savings rate against thousands of data points to generate an accurate, real-time Financial Health Score.</p>
              <ul className="story-features">
                <li><span className="material-icons">check_circle</span> Real-time evaluation of spending habits</li>
                <li><span className="material-icons">check_circle</span> Identifies critical risk factors</li>
                <li><span className="material-icons">check_circle</span> Actionable steps to improve your score</li>
              </ul>
            </motion.div>
            <motion.div className="story-visual" variants={fadeInUp}>
              <div className="story-glass-card" style={{ textAlign: 'center', padding: '40px' }}>
                <div style={{ fontSize: '14px', color: 'var(--gray-700)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Current Score</div>
                <div style={{ fontSize: '72px', fontWeight: 700, color: 'var(--primary)', lineHeight: 1, marginBottom: '16px', fontFamily: 'var(--font-heading)' }}>84</div>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', background: 'var(--success-bg)', color: 'var(--success)', padding: '6px 12px', borderRadius: '12px', fontSize: '14px', fontWeight: 600 }}>
                  <span className="material-icons" style={{ fontSize: '16px' }}>trending_up</span> Excellent
                </div>
              </div>
              <div className="floating-element floating-2" style={{ top: '-50px', right: '-50px', background: 'linear-gradient(135deg, var(--success) 0%, var(--primary) 100%)' }}></div>
            </motion.div>
          </motion.div>

          {/* Story 2: SHAP Explainable AI (Reverse) */}
          <motion.div 
            className="story-section reverse"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            style={{ marginTop: '80px' }}
          >
            <motion.div className="story-content" variants={fadeInUp}>
              <div className="story-badge">
                <span className="material-icons">insights</span>
                Explainable AI
              </div>
              <h2 className="story-title">Transparent AI you can trust.</h2>
              <p className="story-description">Never wonder why the AI gave you a certain score. Nexus uses SHAP (SHapley Additive exPlanations) to break down exactly which factors are helping or hurting your financial health.</p>
              <ul className="story-features">
                <li><span className="material-icons">check_circle</span> Understand the "Why" behind the score</li>
                <li><span className="material-icons">check_circle</span> Visualize positive and negative impactors</li>
                <li><span className="material-icons">check_circle</span> Eliminate black-box AI anxiety</li>
              </ul>
            </motion.div>
            <motion.div className="story-visual" variants={fadeInUp}>
              <div className="story-glass-card">
                <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '16px' }}>Key Score Impactors</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '14px', color: 'var(--gray-700)' }}>High Savings Rate</span>
                    <span style={{ color: 'var(--success)', fontWeight: 600, fontSize: '14px' }}>+12 pts</span>
                  </div>
                  <div style={{ width: '100%', height: '6px', background: 'var(--gray-200)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: '80%', height: '100%', background: 'var(--success)' }}></div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
                    <span style={{ fontSize: '14px', color: 'var(--gray-700)' }}>High Debt Utilization</span>
                    <span style={{ color: 'var(--danger)', fontWeight: 600, fontSize: '14px' }}>-8 pts</span>
                  </div>
                  <div style={{ width: '100%', height: '6px', background: 'var(--gray-200)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: '50%', height: '100%', background: 'var(--danger)' }}></div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Story 3: Predictive Forecasting */}
          <motion.div 
            className="story-section"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            style={{ marginTop: '80px' }}
          >
            <motion.div className="story-content" variants={fadeInUp}>
              <div className="story-badge">
                <span className="material-icons">timeline</span>
                Predictive Forecasting
              </div>
              <h2 className="story-title">See your future net worth today.</h2>
              <p className="story-description">Our linear regression models analyze your historical spending and saving velocity to accurately project your wealth trajectory 3, 6, and 12 months into the future.</p>
              <ul className="story-features">
                <li><span className="material-icons">check_circle</span> Trend analysis based on real habits</li>
                <li><span className="material-icons">check_circle</span> Custom milestone goal tracking</li>
                <li><span className="material-icons">check_circle</span> Automated anomaly detection alerts</li>
              </ul>
            </motion.div>
            <motion.div className="story-visual" variants={fadeInUp}>
              <div className="story-glass-card" style={{ height: '220px', display: 'flex', alignItems: 'flex-end', gap: '8px', padding: '24px 12px' }}>
                {/* Mock Chart bars */}
                {[40, 55, 45, 60, 80, 95].map((height, i) => (
                  <div key={i} style={{ 
                    flex: 1, 
                    height: `${height}%`, 
                    background: i > 3 ? 'rgba(67, 97, 238, 0.4)' : 'var(--primary)', 
                    borderRadius: '4px 4px 0 0',
                    border: i > 3 ? '1px dashed var(--primary)' : 'none'
                  }}></div>
                ))}
              </div>
              <div className="floating-element floating-1" style={{ top: '50px', right: '100px' }}></div>
            </motion.div>
          </motion.div>

          {/* Story 4: Nexus AI Assistant (Reverse) */}
          <motion.div 
            className="story-section reverse"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            style={{ marginTop: '80px' }}
          >
            <motion.div className="story-content" variants={fadeInUp}>
              <div className="story-badge">
                <span className="material-icons">smart_toy</span>
                Nexus AI Assistant
              </div>
              <h2 className="story-title">Your personal CFO, available 24/7.</h2>
              <p className="story-description">Chat naturally with Nexus AI. It understands exactly which page you're viewing, knows your financial context, and provides deeply personalized advice powered by Gemini.</p>
              <ul className="story-features">
                <li><span className="material-icons">check_circle</span> Context-aware screen intelligence</li>
                <li><span className="material-icons">check_circle</span> Instant portfolio analysis</li>
                <li><span className="material-icons">check_circle</span> Tailored investment recommendations</li>
              </ul>
            </motion.div>
            <motion.div className="story-visual" variants={fadeInUp} style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ background: '#ffffff', border: '1px solid var(--border)', borderRadius: '24px', display: 'flex', flexDirection: 'column', height: '320px' }}>
                <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--gray-200)', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                  <span className="material-icons" style={{ color: 'var(--primary)' }}>auto_awesome</span> Nexus AI
                </div>
                <div style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', background: 'var(--gray-100)' }}>
                  <div style={{ alignSelf: 'flex-end', background: 'var(--primary)', color: 'white', padding: '10px 14px', borderRadius: '12px 12px 0 12px', fontSize: '13px' }}>
                    Should I invest in tech stocks?
                  </div>
                  <div style={{ alignSelf: 'flex-start', background: 'white', border: '1px solid var(--gray-200)', color: 'var(--dark)', padding: '12px 16px', borderRadius: '12px 12px 12px 0', fontSize: '13px', boxShadow: '0 4px 12px rgba(0,0,0,0.02)' }}>
                    Based on your <strong>High</strong> risk profile and current <strong>60% Equity</strong> allocation, a diversified tech index fits well. I recommend viewing the Growth Mutual Fund section.
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>

        </div>
      </section>
      
      <section className="cta-container">
        <div className="container" style={{ display: 'flex', justifyContent: 'center' }}>
          <motion.div 
            className="cta-card"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeInUp}
          >
            <h2 className="cta-title">Elevate Your Wealth Strategy</h2>
            <p className="cta-subtitle">Join the waitlist for the next generation of autonomous finance. Make smarter, data-driven decisions with Nexus AI.</p>
            
            <div className="cta-actions">
              <button className="btn btn-primary" onClick={() => navigate('/login#signup')}>Get Started Free</button>
              <button className="btn btn-ghost" onClick={() => {
                useDemoStore.getState().startDemo();
                navigate('/dashboard');
              }}>Explore Demo</button>
            </div>
            
            <div className="cta-trust">
              <span className="material-icons">lock</span> Bank-level 256-bit encryption. We never sell your data.
            </div>
          </motion.div>
        </div>
      </section>
      
      <footer className="landing-footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="footer-logo">
                <span className="material-icons">auto_awesome</span>
                Nexus Finance AI
              </div>
              <p className="footer-description">Advanced financial intelligence powered by artificial intelligence to help you make smarter decisions. Build wealth with confidence.</p>
              
              <div className="social-links">
                <a href="#" className="social-link" aria-label="Twitter">
                  <i className="fab fa-twitter"></i>
                </a>
                <a href="#" className="social-link" aria-label="LinkedIn">
                  <i className="fab fa-linkedin-in"></i>
                </a>
                <a href="#" className="social-link" aria-label="GitHub">
                  <i className="fab fa-github"></i>
                </a>
              </div>
            </div>
            
            <div className="footer-nav">
              <h3 className="footer-title">Product</h3>
              <ul className="footer-links">
                <li><a href="#features">Financial Health</a></li>
                <li><a href="#solutions">Spending Forecasts</a></li>
                <li><a href="#solutions">Investment Planning</a></li>
                <li><a href="#solutions">Security</a></li>
              </ul>
            </div>
            
            <div className="footer-nav">
              <h3 className="footer-title">Technology</h3>
              <ul className="footer-links">
                <li><a href="#resources">SHAP Explainability</a></li>
                <li><a href="#resources">Anomaly Detection</a></li>
                <li><a href="#resources">Random Forest Models</a></li>
                <li><a href="#resources">API Docs</a></li>
              </ul>
            </div>

            <div className="footer-nav">
              <h3 className="footer-title">Nexus AI</h3>
              <ul className="footer-links">
                <li><a href="#resources">Gemini Integration</a></li>
                <li><a href="#resources">Context Awareness</a></li>
                <li><a href="#resources">Personal CFO</a></li>
              </ul>
            </div>
            
            <div className="footer-nav">
              <h3 className="footer-title">Company</h3>
              <ul className="footer-links">
                <li><a href="#company">About Us</a></li>
                <li><a href="#company">Privacy Policy</a></li>
                <li><a href="#company">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          
          <div className="footer-bottom">
            <p>&copy; 2026 Nexus Finance AI. All rights reserved.</p>
            <div style={{ display: 'flex', gap: '20px' }}>
              <a href="#" style={{ color: 'var(--gray-700)', textDecoration: 'none' }}>Privacy</a>
              <a href="#" style={{ color: 'var(--gray-700)', textDecoration: 'none' }}>Terms</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
