import { create } from 'zustand';
import { api } from '../services/api';

export interface UserProfile {
  id: number;
  email: string;
  name: string;
  income: number | null;
  goal: string | null;
  risk_profile: string | null;
}

interface AuthState {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (token: string) => Promise<void>;
  logout: () => void;
  fetchProfile: () => Promise<UserProfile | null>;
  setProfile: (profile: Partial<UserProfile>) => void;
}

// Utility to parse token expiration from JWT
const isTokenExpired = (token: string): boolean => {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    if (payload.exp && Date.now() >= payload.exp * 1000) {
      return true;
    }
    return false;
  } catch (e) {
    return true;
  }
};

export const useAuthStore = create<AuthState>((set, get) => {
  const initialToken = localStorage.getItem('jwt_token');
  const isAuth = !!initialToken && !isTokenExpired(initialToken);

  return {
    user: null,
    isAuthenticated: isAuth,
    isLoading: false,

    login: async (token: string) => {
      localStorage.setItem('jwt_token', token);
      const user = await get().fetchProfile();
      if (user) {
        set({ isAuthenticated: true, user });
      } else {
        set({ isAuthenticated: false, user: null });
      }
    },

    logout: () => {
      localStorage.removeItem('jwt_token');
      set({ isAuthenticated: false, user: null });
    },

    fetchProfile: async () => {
      try {
        set({ isLoading: true });
        const response = await api.get('/auth/me');
        const userData = response.data.user;
        set({ user: userData, isAuthenticated: true, isLoading: false });
        return userData;
      } catch (error) {
        set({ user: null, isAuthenticated: false, isLoading: false });
        return null;
      }
    },

    setProfile: (profile: Partial<UserProfile>) => {
      set((state) => ({
        user: state.user ? { ...state.user, ...profile } : null,
      }));
    },
  };
});
