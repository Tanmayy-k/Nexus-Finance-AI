import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { useDemoStore } from '../store/useDemoStore';
import { demoData } from '../store/demoData';

export interface StockSymbolList {
  available_symbols: string[];
}

export interface StockDataPoint {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Symbol: string;
}

// 1. Fetch available stock symbols list
export const useStocksList = () => {
  return useQuery({
    queryKey: ['stocks_list'],
    queryFn: async (): Promise<string[]> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.stocks_list.available_symbols;
      }
      const response = await api.get('/api/stocks/list');
      return response.data.available_symbols || [];
    }
  });
};

// 2. Fetch stock data for a given symbol
export const useStockData = (symbol: string) => {
  return useQuery({
    queryKey: ['stock_data', symbol],
    queryFn: async (): Promise<StockDataPoint[]> => {
      if (useDemoStore.getState().isDemoMode) {
        await new Promise(resolve => setTimeout(resolve, 400));
        return demoData.stock_data as StockDataPoint[];
      }
      const response = await api.get(`/api/stock_data/${symbol}`);
      return response.data;
    },
    enabled: !!symbol
  });
};
