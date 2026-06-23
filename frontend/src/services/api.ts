import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor: Attach JWT Token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('jwt_token');
    if (token && config.headers) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response Interceptor: Handle 401 Session Expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('jwt_token');
      // Dispatch custom event to trigger React state updates or redirection
      window.dispatchEvent(new Event('auth-unauthorized'));
    }
    return Promise.reject(error);
  }
);
import { useDemoStore } from '../store/useDemoStore';
import { demoData } from '../store/demoData';

// Delay helper to simulate network
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const originalGet = api.get;
api.get = async function(url: string, config?: any) {
  if (useDemoStore.getState().isDemoMode) {
    await delay(300); // Realistic loading
    
    if (url === '/summary') return { data: demoData.summary };
    if (url === '/expenses') return { data: demoData.expenses };
    if (url === '/anomalies') return { data: demoData.anomalies };
    if (url === '/forecast') return { data: demoData.forecast };
    if (url === '/insights') return { data: demoData.insights };
    if (url === '/weekly_focus') return { data: demoData.weekly_focus };
    if (url.startsWith('/recommendations')) return { data: demoData.recommendations };
    if (url === '/portfolio/valuation') return { data: demoData.portfolio_valuation };
    if (url === '/budget') return { data: demoData.budget };
    if (url === '/tips') return { data: demoData.tips };
    if (url === '/cluster_users') return { data: demoData.cluster_users };
    if (url === '/api/stocks/list') return { data: demoData.stocks_list };
    if (url.startsWith('/api/stock_data')) return { data: demoData.stock_data };
    if (url === '/user/profile') return { data: { name: 'Demo User', email: 'demo@nexus.ai', income: 80000, goal: 'Retirement', risk_profile: 'high' } };
  }
  return originalGet.apply(this, [url, config]);
};

const originalPost = api.post;
api.post = async function(url: string, data?: any, config?: any) {
  if (useDemoStore.getState().isDemoMode) {
    await delay(400); // Realistic loading
    if (url === '/predict') return { data: demoData.predict };
    if (url === '/copilot/chat') return { data: demoData.copilot_chat };
  }
  return originalPost.apply(this, [url, data, config]);
};
