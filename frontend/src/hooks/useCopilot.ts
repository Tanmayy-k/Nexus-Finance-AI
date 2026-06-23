import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../services/api';
import { useDemoStore } from '../store/useDemoStore';
import { demoData } from '../store/demoData';

export interface ScoreFactors {
  positive: Array<{ label: string; impact: number }>;
  negative: Array<{ label: string; impact: number }>;
}

export interface PredictionData {
  score: number;
  prediction: number;
  factors: ScoreFactors;
}

export const useCopilotStatus = () => {
  return useQuery({
    queryKey: ['copilot_status'],
    queryFn: async () => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return { mode: 'gemini' };
      }
      const response = await api.get('/copilot/status');
      return response.data; // { mode: 'gemini' | 'fallback' }
    }
  });
};

export const useCopilotChat = () => {
  return useMutation({
    mutationFn: async (payload: { message: string; current_page: string }) => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return { reply: demoData.copilot_chat.reply, mode: 'gemini' };
      }
      const response = await api.post('/copilot/chat', payload);
      return response.data; // { reply: string, mode: 'gemini' | 'fallback' }
    }
  });
};

export const usePredictScore = () => {
  return useMutation({
    mutationFn: async (payload: { monthly_income: number; monthly_expense_total: number; investment_amount: number }): Promise<PredictionData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.predict as PredictionData;
      }
      const response = await api.post('/predict', payload);
      return response.data;
    }
  });
};
