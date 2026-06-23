import React from 'react';
import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
  title?: string;
  duration?: number;
}

interface ToastState {
  toasts: ToastItem[];
  addToast: (message: string, type?: ToastType, options?: { title?: string; duration?: number }) => void;
  removeToast: (id: string) => void;
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  addToast: (message, type = 'info', options = {}) => {
    const id = 'toast_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    const duration = options.duration ?? (type === 'error' ? 6000 : type === 'warning' ? 5000 : 4000);
    
    set((state) => ({
      toasts: [...state.toasts, { id, message, type, title: options.title, duration }],
    }));

    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, duration);
  },
  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },
}));

// Expose globals for backward compatibility with inline scripts
if (typeof window !== 'undefined') {
  (window as any).showToast = (message: string, type: ToastType = 'info', options: any = {}) => {
    useToastStore.getState().addToast(message, type, options);
  };
  (window as any).showSuccess = (message: string, options: any = {}) => {
    useToastStore.getState().addToast(message, 'success', options);
  };
  (window as any).showError = (message: string, options: any = {}) => {
    useToastStore.getState().addToast(message, 'error', options);
  };
  (window as any).showWarning = (message: string, options: any = {}) => {
    useToastStore.getState().addToast(message, 'warning', options);
  };
  (window as any).showInfo = (message: string, options: any = {}) => {
    useToastStore.getState().addToast(message, 'info', options);
  };
}

export const ToastContainer: React.FC = () => {
  const toasts = useToastStore((state) => state.toasts);
  const removeToast = useToastStore((state) => state.removeToast);

  return (
    <div className="notification-container">
      {toasts.map((toast) => {
        const iconMap = {
          success: '✓',
          error: '✕',
          warning: '⚠',
          info: 'ℹ',
        };
        const titleMap = {
          success: 'Success',
          error: 'Error',
          warning: 'Warning',
          info: 'Info',
        };

        return (
          <div key={toast.id} className={`notification ${toast.type} show`} onClick={() => removeToast(toast.id)}>
            <div className="notification-icon">{iconMap[toast.type]}</div>
            <div className="notification-content">
              <div className="notification-title">{toast.title || titleMap[toast.type]}</div>
              <div className="notification-message">{toast.message}</div>
            </div>
            <button className="notification-close" onClick={(e) => {
              e.stopPropagation();
              removeToast(toast.id);
            }}>×</button>
            <div 
              className="notification-progress" 
              style={{ animation: `progress ${toast.duration}ms linear forwards` }}
            />
          </div>
        );
      })}
    </div>
  );
};
