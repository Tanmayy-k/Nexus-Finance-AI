import { create } from 'zustand';

interface DemoState {
  isDemoMode: boolean;
  isTourActive: boolean;
  currentTourStep: number;
  startDemo: () => void;
  exitDemo: () => void;
  startTour: () => void;
  endTour: () => void;
  nextTourStep: () => void;
  prevTourStep: () => void;
}

export const useDemoStore = create<DemoState>((set) => ({
  isDemoMode: false,
  isTourActive: false,
  currentTourStep: 0,
  
  startDemo: () => set({ isDemoMode: true, isTourActive: true, currentTourStep: 0 }),
  exitDemo: () => set({ isDemoMode: false, isTourActive: false, currentTourStep: 0 }),
  
  startTour: () => set({ isTourActive: true, currentTourStep: 0 }),
  endTour: () => set({ isTourActive: false }),
  
  nextTourStep: () => set((state) => ({ currentTourStep: state.currentTourStep + 1 })),
  prevTourStep: () => set((state) => ({ currentTourStep: Math.max(0, state.currentTourStep - 1) })),
}));
