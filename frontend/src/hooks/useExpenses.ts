import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../services/api';
import { useDemoStore } from '../store/useDemoStore';
import { demoData } from '../store/demoData';

export interface ExpenseItem {
  id: number;
  category: string;
  amount: number;
  description: string;
  date: string;
}

export interface SummaryData {
  total_spent: number;
  by_category: Record<string, number>;
  alerts: string[];
}

export interface AnomalyItem {
  id: number;
  category: string;
  amount: number;
  date: string;
  description: string;
  anomaly_score: number;
  reason: string;
}

export interface AnomalyData {
  has_data: boolean;
  total_analyzed: number;
  anomalies: AnomalyItem[];
  message: string;
}

export interface ForecastData {
  has_data: boolean;
  months_available: number;
  current_month_total: number;
  next_month_forecast: number;
  change_pct: number;
  trend: 'increasing' | 'decreasing' | 'stable' | 'unknown';
  message: string;
  history?: Array<{ month: string; amount: number }>;
  forecast?: Array<{ month: string; predicted_amount: number }>;
}

export interface InsightItem {
  type: string;
  severity: 'warning' | 'positive' | 'info';
  message: string;
}

export interface InsightsData {
  insights: InsightItem[];
  summary: string;
}

export interface WeeklyFocusData {
  focus_message: string;
}

// 1. Fetch Expenses List
export const useExpenses = () => {
  return useQuery({
    queryKey: ['expenses'],
    queryFn: async (): Promise<ExpenseItem[]> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.expenses;
      }
      const response = await api.get('/expenses');
      return response.data.expenses || [];
    }
  });
};

// 2. Fetch Financial Summary
export const useFinancialSummary = () => {
  return useQuery({
    queryKey: ['summary'],
    queryFn: async (): Promise<SummaryData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.summary;
      }
      const response = await api.get('/summary');
      return response.data;
    }
  });
};

// 3. Fetch Anomalies
export const useAnomalies = () => {
  return useQuery({
    queryKey: ['anomalies'],
    queryFn: async (): Promise<AnomalyData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.anomalies;
      }
      const response = await api.get('/anomalies');
      return response.data;
    }
  });
};

// 4. Fetch Spending Forecast
export const useForecast = () => {
  return useQuery({
    queryKey: ['forecast'],
    queryFn: async (): Promise<ForecastData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.forecast as ForecastData;
      }
      const response = await api.get('/forecast');
      return response.data;
    }
  });
};

// 5. Fetch Insights
export const useInsights = () => {
  return useQuery({
    queryKey: ['insights'],
    queryFn: async (): Promise<InsightsData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.insights as InsightsData;
      }
      const response = await api.get('/insights');
      return response.data;
    }
  });
};

// 6. Fetch Weekly Focus
export const useWeeklyFocus = () => {
  return useQuery({
    queryKey: ['weekly_focus'],
    queryFn: async (): Promise<WeeklyFocusData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.weekly_focus;
      }
      const response = await api.get('/weekly_focus');
      return response.data;
    }
  });
};

// 7. Add Expense Mutation
export const useAddExpense = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: { category: string; amount: number; date: string; description: string }) => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return { success: true };
      }
      const response = await api.post('/expense', payload);
      return response.data;
    },
    onSuccess: () => {
      // Invalidate queries to refresh dashboard components in real-time
      queryClient.invalidateQueries({ queryKey: ['expenses'] });
      queryClient.invalidateQueries({ queryKey: ['summary'] });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      queryClient.invalidateQueries({ queryKey: ['forecast'] });
      queryClient.invalidateQueries({ queryKey: ['insights'] });
      queryClient.invalidateQueries({ queryKey: ['weekly_focus'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio_valuation'] });
      queryClient.invalidateQueries({ queryKey: ['budget_tips'] });
    }
  });
};

export interface BudgetTipResponse {
  tip: string;
}

// 8. Delete Expense Mutation
export const useDeleteExpense = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: number | string) => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return { success: true };
      }
      const response = await api.delete(`/expense/${id}`);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['expenses'] });
      queryClient.invalidateQueries({ queryKey: ['summary'] });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      queryClient.invalidateQueries({ queryKey: ['forecast'] });
      queryClient.invalidateQueries({ queryKey: ['insights'] });
      queryClient.invalidateQueries({ queryKey: ['weekly_focus'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio_valuation'] });
      queryClient.invalidateQueries({ queryKey: ['budget_tips'] });
    }
  });
};

// 9. Fetch Budget Tips
export const useBudgetTips = () => {
  return useQuery({
    queryKey: ['budget_tips'],
    queryFn: async (): Promise<BudgetTipResponse> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return { tip: demoData.tips.tips[0] };
      }
      const response = await api.get('/tips');
      return response.data;
    }
  });
};
