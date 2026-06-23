import { create } from 'zustand';
import { api } from '../services/api';

export interface BudgetSplit {
  Housing: number;
  Food: number;
  Transportation: number;
  Utilities: number;
  Entertainment: number;
  Savings: number;
  [key: string]: number;
}

interface BudgetState {
  hasBudget: boolean;
  income: number | null;
  goal: string | null;
  budgetSplit: BudgetSplit | null;
  isLoading: boolean;
  fetchBudget: () => Promise<void>;
  createOrUpdateBudget: (income: number, goal: string, riskProfile: string) => Promise<void>;
  resetBudget: () => void;
}

export const useBudgetStore = create<BudgetState>((set) => ({
  hasBudget: false,
  income: null,
  goal: null,
  budgetSplit: null,
  isLoading: false,

  fetchBudget: async () => {
    set({ isLoading: true });
    try {
      const response = await api.get('/budget');
      if (response.data.has_budget) {
        set({
          hasBudget: true,
          income: response.data.income,
          goal: response.data.goal,
          budgetSplit: response.data.budget_split,
        });
      } else {
        set({ hasBudget: false, budgetSplit: null });
      }
    } catch (error) {
      console.error('Failed to fetch budget:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  createOrUpdateBudget: async (income: number, goal: string, riskProfile: string) => {
    set({ isLoading: true });
    try {
      const response = await api.post('/budget', {
        income,
        goal,
        risk_profile: riskProfile,
      });
      if (response.data.has_budget) {
        set({
          hasBudget: true,
          income: response.data.income,
          goal: response.data.goal,
          budgetSplit: response.data.budget_split,
        });
      }
    } catch (error) {
      console.error('Failed to create/update budget:', error);
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  resetBudget: () => {
    set({ hasBudget: false, income: null, goal: null, budgetSplit: null });
  },
}));
