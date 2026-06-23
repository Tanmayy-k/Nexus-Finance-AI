import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../services/api';
import { useDemoStore } from '../store/useDemoStore';
import { demoData } from '../store/demoData';

export interface RecommendationItem {
  scheme: string;
  reason: string;
}

export interface RecommendationsData {
  user_id: number;
  cluster: number | null;
  recommendations: RecommendationItem[];
}

export interface ClusterUser {
  id: number;
  income: number;
  goal: number;
  risk: number;
  cluster: number;
}

export interface ClusteringData {
  clusters: ClusterUser[];
  current_user_cluster: number;
}

export interface InvestmentSuggestionsPayload {
  risk: string;
  savings: number;
}

export interface InvestmentSuggestionsData {
  risk_profile: string;
  savings: number;
  suggestions: string[];
}

// 1. Fetch Tailored Recommendations
export const useRecommendations = () => {
  return useQuery({
    queryKey: ['recommendations'],
    queryFn: async (): Promise<RecommendationsData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return {
          user_id: 1,
          cluster: 1,
          recommendations: demoData.recommendations.recommendations
        };
      }
      const response = await api.get('/recommendations?n=6');
      return response.data;
    }
  });
};

// 2. Fetch User Clustering Analytics
export const useUserClustering = () => {
  return useQuery({
    queryKey: ['user_clustering'],
    queryFn: async (): Promise<ClusteringData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.cluster_users as ClusteringData;
      }
      const response = await api.get('/cluster_users');
      return response.data;
    }
  });
};

// 3. Query Investment Suggestions by Risk/Savings
export const useInvestmentSuggestions = () => {
  return useMutation({
    mutationFn: async (payload: InvestmentSuggestionsPayload): Promise<InvestmentSuggestionsData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return {
          risk_profile: payload.risk,
          savings: payload.savings,
          suggestions: demoData.recommendations.recommendations.map((r: any) => r.scheme)
        };
      }
      const response = await api.post('/investment', payload);
      return response.data;
    }
  });
};

export interface PortfolioValuationData {
  total_investment: number;
  current_value: number;
  absolute_gain: number;
  gain_percentage: number;
  cash_available: number;
  net_worth: number;
}

// 4. Fetch Portfolio Valuation
export const usePortfolioValuation = () => {
  return useQuery({
    queryKey: ['portfolio_valuation'],
    queryFn: async (): Promise<PortfolioValuationData> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return {
          total_investment: demoData.portfolio_valuation.total_invested,
          current_value: demoData.portfolio_valuation.current_value,
          absolute_gain: demoData.portfolio_valuation.absolute_gain,
          gain_percentage: demoData.portfolio_valuation.gain_percentage,
          cash_available: demoData.portfolio_valuation.cash_available,
          net_worth: demoData.portfolio_valuation.net_worth
        };
      }
      const response = await api.get('/portfolio/valuation');
      return response.data;
    }
  });
};
