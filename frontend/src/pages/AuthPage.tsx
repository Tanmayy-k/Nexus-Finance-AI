import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/useAuthStore';
import { useDemoStore } from '../store/useDemoStore';
import { api } from '../services/api';
import '../styles/auth.css';

export const AuthPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login: storeLogin, isAuthenticated } = useAuthStore();

  const [activeTab, setActiveTab] = useState<'login' | 'signup'>('login');
  
  // Login states
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // Signup states
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [showSignupPassword, setShowSignupPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSigningUp, setIsSigningUp] = useState(false);

  useEffect(() => {
    // If user is already authenticated, send them to dashboard
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    // Always exit demo mode if we land on the Auth page
    // (Import is placed at the top of the file)
    useDemoStore.getState().exitDemo();
    
    // Check hash for #signup
    if (location.hash === '#signup') {
      setActiveTab('signup');
    } else {
      setActiveTab('login');
    }
  }, [location.hash]);

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoggingIn) return;
    setIsLoggingIn(true);

    try {
      const response = await api.post('/auth/login', {
        email: loginEmail,
        password: loginPassword,
      });

      const data = response.data;
      if (data.token) {
        await storeLogin(data.token);
        (window as any).showToast('Login successful', 'success');
        navigate('/dashboard');
      } else {
        throw new Error(data.error || 'Login failed');
      }
    } catch (error: any) {
      const errMsg = error.response?.data?.error || error.message || 'Login failed';
      (window as any).showToast(errMsg, 'error');
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleSignupSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSigningUp) return;

    if (signupPassword !== confirmPassword) {
      (window as any).showToast('Passwords do not match!', 'error');
      return;
    }

    if (!agreeTerms) {
      (window as any).showToast('You must agree to the terms', 'warning');
      return;
    }

    setIsSigningUp(true);

    try {
      const response = await api.post('/auth/register', {
        name: signupName,
        email: signupEmail,
        password: signupPassword,
      });

      const data = response.data;
      if (response.status === 201 || data.message) {
        (window as any).showToast('Account created! Logging you in...', 'success');
        
        // Auto-login right after signup
        try {
          const loginRes = await api.post('/auth/login', {
            email: signupEmail,
            password: signupPassword,
          });
          if (loginRes.data.token) {
            await storeLogin(loginRes.data.token);
            navigate('/dashboard');
          }
        } catch (loginErr) {
          (window as any).showToast('Please log in manually', 'warning');
          setActiveTab('login');
        }
      }
    } catch (error: any) {
      const errMsg = error.response?.data?.error || error.message || 'Registration failed';
      (window as any).showToast(errMsg, 'error');
    } finally {
      setIsSigningUp(false);
    }
  };

  return (
    <div className="auth-body">
      <div className="header" style={{ position: 'absolute', top: '30px', left: '30px', right: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 0 }}>
        <button 
          className="btn btn-ghost" 
          onClick={() => navigate('/')} 
          style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px' }}
        >
          <span className="material-icons">arrow_back</span>
          Back
        </button>
        <div className="logo" style={{ cursor: 'pointer', margin: '0 auto', position: 'absolute', left: '50%', transform: 'translateX(-50%)' }} onClick={() => navigate('/')}>
          <div className="logo-icon">
            <span className="material-icons">auto_awesome</span>
          </div>
          <div className="logo-text">Nexus Finance AI</div>
        </div>
      </div>

      <div className="auth-container">
        <div className="form-tabs">
          <div 
            className={`form-tab ${activeTab === 'login' ? 'active' : ''}`} 
            onClick={() => setActiveTab('login')}
          >
            Sign In
          </div>
          <div 
            className={`form-tab ${activeTab === 'signup' ? 'active' : ''}`} 
            onClick={() => setActiveTab('signup')}
          >
            Create Account
          </div>
        </div>
        
        {/* Login Form */}
        <form 
          id="login-form" 
          className={`auth-form ${activeTab === 'login' ? 'active' : ''}`} 
          onSubmit={handleLoginSubmit}
        >
          <h3 className="form-title">Welcome back</h3>
          <p className="form-subtitle">Sign in to access your dashboard</p>
          
          <div className="form-group">
            <label htmlFor="login-email" className="form-label">Email Address</label>
            <input 
              type="email" 
              id="login-email" 
              className="form-input" 
              placeholder="name@company.com" 
              value={loginEmail}
              onChange={(e) => setLoginEmail(e.target.value)}
              required 
            />
          </div>
          
          <div className="form-group password-toggle">
            <label htmlFor="login-password" className="form-label">Password</label>
            <input 
              type={showLoginPassword ? 'text' : 'password'} 
              id="login-password" 
              className="form-input" 
              placeholder="••••••••" 
              value={loginPassword}
              onChange={(e) => setLoginPassword(e.target.value)}
              required 
            />
            <button 
              type="button" 
              className="toggle-password" 
              onClick={() => setShowLoginPassword(!showLoginPassword)}
              aria-label="Toggle password visibility"
            >
              <span className="material-icons">
                {showLoginPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
          
          <div className="form-options">
            <label className="remember-me">
              <input type="checkbox" id="remember-me" />
              <span>Remember me</span>
            </label>
            <a href="#" className="forgot-password" onClick={(e) => e.preventDefault()}>Forgot password?</a>
          </div>
          
          <button type="submit" className="btn btn-primary" disabled={isLoggingIn}>
            {isLoggingIn ? 'Signing In...' : 'Sign In'}
          </button>
          
          <div className="auth-divider">
            <span>Or continue with</span>
          </div>
          
          <div className="social-auth">
            <button type="button" className="social-btn" onClick={() => (window as any).showToast('Google login coming soon', 'info')}>
              <span className="material-icons" style={{ color: '#DB4437' }}>mail</span>
              Google
            </button>
            <button type="button" className="social-btn" onClick={() => (window as any).showToast('Apple login coming soon', 'info')}>
              <span className="material-icons" style={{ color: '#000' }}>phone_iphone</span>
              Apple
            </button>
          </div>
          
          <div className="auth-switch">
            <p>Don't have an account? <span className="auth-link" onClick={() => setActiveTab('signup')}>Sign up</span></p>
          </div>
        </form>
        
        {/* Signup Form */}
        <form 
          id="signup-form" 
          className={`auth-form ${activeTab === 'signup' ? 'active' : ''}`} 
          onSubmit={handleSignupSubmit}
        >
          <h3 className="form-title">Create account</h3>
          <p className="form-subtitle">Get started with your free account</p>
          
          <div className="form-group">
            <label htmlFor="signup-name" className="form-label">Full Name</label>
            <input 
              type="text" 
              id="signup-name" 
              className="form-input" 
              placeholder="John Doe" 
              value={signupName}
              onChange={(e) => setSignupName(e.target.value)}
              required 
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="signup-email" className="form-label">Email Address</label>
            <input 
              type="email" 
              id="signup-email" 
              className="form-input" 
              placeholder="name@company.com" 
              value={signupEmail}
              onChange={(e) => setSignupEmail(e.target.value)}
              required 
            />
          </div>
          
          <div className="form-group password-toggle">
            <label htmlFor="signup-password" className="form-label">Password</label>
            <input 
              type={showSignupPassword ? 'text' : 'password'} 
              id="signup-password" 
              className="form-input" 
              placeholder="••••••••" 
              value={signupPassword}
              onChange={(e) => setSignupPassword(e.target.value)}
              required 
            />
            <button 
              type="button" 
              className="toggle-password" 
              onClick={() => setShowSignupPassword(!showSignupPassword)}
              aria-label="Toggle password visibility"
            >
              <span className="material-icons">
                {showSignupPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
          
          <div className="form-group password-toggle">
            <label htmlFor="confirm-password" className="form-label">Confirm Password</label>
            <input 
              type={showConfirmPassword ? 'text' : 'password'} 
              id="confirm-password" 
              className="form-input" 
              placeholder="••••••••" 
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required 
            />
            <button 
              type="button" 
              className="toggle-password" 
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              aria-label="Toggle password visibility"
            >
              <span className="material-icons">
                {showConfirmPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
          
          <div className="form-options">
            <label className="remember-me">
              <input 
                type="checkbox" 
                id="terms-agree" 
                checked={agreeTerms}
                onChange={(e) => setAgreeTerms(e.target.checked)}
                required 
              />
              <span>I agree to the <a href="#" className="auth-link" onClick={(e) => e.preventDefault()}>Terms & Privacy</a></span>
            </label>
          </div>
          
          <button type="submit" className="btn btn-primary" disabled={isSigningUp}>
            {isSigningUp ? 'Creating Account...' : 'Create Account'}
          </button>
          
          <div className="auth-divider">
            <span>Or continue with</span>
          </div>
          
          <div className="social-auth">
            <button type="button" className="social-btn" onClick={() => (window as any).showToast('Google registration coming soon', 'info')}>
              <span className="material-icons" style={{ color: '#DB4437' }}>mail</span>
              Google
            </button>
            <button type="button" className="social-btn" onClick={() => (window as any).showToast('Apple registration coming soon', 'info')}>
              <span className="material-icons" style={{ color: '#000' }}>phone_iphone</span>
              Apple
            </button>
          </div>
          
          <div className="auth-switch">
            <p>Already have an account? <span className="auth-link" onClick={() => setActiveTab('login')}>Sign in</span></p>
          </div>
        </form>
      </div>
      
      {/* Footer */}
      <div className="footer">
        <div>© 2023 Nexus Finance AI. All rights reserved.</div>
        <div className="footer-links">
          <a href="#" className="footer-link" onClick={(e) => e.preventDefault()}>Privacy Policy</a>
          <a href="#" className="footer-link" onClick={(e) => e.preventDefault()}>Terms of Service</a>
          <a href="#" className="footer-link" onClick={(e) => e.preventDefault()}>Contact Us</a>
        </div>
      </div>
    </div>
  );
};
