import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/useAuthStore';

interface ProtectedRouteProps {
  children: React.ReactElement;
}

import { useDemoStore } from '../store/useDemoStore';

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, fetchProfile, user } = useAuthStore();
  const isDemoMode = useDemoStore((state) => state.isDemoMode);

  useEffect(() => {
    if (isAuthenticated && !user) {
      fetchProfile();
    }
  }, [isAuthenticated, user, fetchProfile]);

  if (!isAuthenticated && !isDemoMode) {
    return <Navigate to="/login" replace />;
  }

  return children;
};
