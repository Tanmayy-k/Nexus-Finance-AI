import React from 'react';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
);

// Global Chart.js Defaults / Helper Styles
const chartFont = {
  family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  size: 11
};

const commonTooltip = {
  backgroundColor: '#0f172a',
  titleColor: '#ffffff',
  bodyColor: '#cbd5e1',
  borderColor: '#334155',
  borderWidth: 1,
  padding: 10,
  cornerRadius: 6,
  boxPadding: 4,
  usePointStyle: true,
  titleFont: { ...chartFont, weight: 'bold' as const },
  bodyFont: chartFont
};

const commonGridX = {
  display: false
};

const commonGridY = {
  color: '#f1f5f9',
  drawBorder: false
};

const commonTicks = {
  color: '#64748b',
  font: chartFont
};

// --- 1. NET WORTH CHART ---
interface NetWorthChartProps {
  income: number;
  totalSpent: number;
}
export const NetWorthChart: React.FC<NetWorthChartProps> = ({ income, totalSpent }) => {
  const baseNetWorth = 0;
  const monthlySavings = Math.max(0, income - totalSpent);
  
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const currentMonthIdx = new Date().getMonth();
  
  const netWorthData = months.map((_, i) => {
    if (i > currentMonthIdx) return null;
    return baseNetWorth + (i * monthlySavings);
  });
  
  const investmentData = months.map((_, i) => {
    if (i > currentMonthIdx) return null;
    return (i * monthlySavings * 0.5);
  });

  const data = {
    labels: months,
    datasets: [
      {
        label: 'Net Worth',
        data: netWorthData,
        borderColor: '#4f46e5', // Primary Indigo
        backgroundColor: 'rgba(79, 70, 229, 0.04)',
        borderWidth: 1.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4
      },
      {
        label: 'Investments',
        data: investmentData,
        borderColor: '#06b6d4', // Secondary Cyan
        backgroundColor: 'rgba(6, 182, 212, 0.04)',
        borderWidth: 1.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 8,
          boxHeight: 8,
          color: '#475569',
          font: { ...chartFont, weight: 'normal' as const }
        }
      },
      tooltip: {
        ...commonTooltip,
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (context: any) => ` ${context.dataset.label}: ₹${context.raw.toLocaleString()}`
        }
      }
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: {
          ...commonTicks,
          callback: (value: any) => '₹' + value.toLocaleString()
        }
      }
    }
  };

  return <Line data={data} options={options} />;
};


// --- 2. MARKET CHART ---
export const MarketChart: React.FC = () => {
  const days = Array.from({ length: 30 }, (_, i) => `${i + 1} Nov`);
  const baseVal = 4500;
  const dataPoints = [baseVal];
  for (let i = 1; i < 30; i++) {
    const change = (Math.random() - 0.45) * 30;
    dataPoints.push(Math.round(dataPoints[i - 1] + change));
  }

  const data = {
    labels: days,
    datasets: [
      {
        label: 'S&P 500',
        data: dataPoints,
        borderColor: '#4f46e5',
        borderWidth: 1.5,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: false
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: commonTooltip
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: commonTicks
      }
    }
  };

  return <Line data={data} options={options} />;
};


// --- 3. ALLOCATION CHART ---
interface AllocationChartProps {
  riskProfile: string;
}
export const AllocationChart: React.FC<AllocationChartProps> = ({ riskProfile }) => {
  let percentages = [40, 20, 15, 10, 10, 5]; // US, Int, Bonds, RE, Cash, Crypto (Moderate)
  
  if (riskProfile.toLowerCase() === 'low') {
    percentages = [15, 5, 50, 5, 25, 0];
  } else if (riskProfile.toLowerCase() === 'high') {
    percentages = [45, 25, 5, 10, 5, 10];
  }

  const data = {
    labels: ['US Stocks', 'Intl Stocks', 'Bonds', 'Real Estate', 'Cash', 'Crypto'],
    datasets: [{
      data: percentages,
      backgroundColor: [
        '#4f46e5', // Indigo
        '#6366f1', // Indigo light
        '#06b6d4', // Cyan
        '#14b8a6', // Teal
        '#94a3b8', // Slate
        '#8b5cf6'  // Violet
      ],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          boxWidth: 8,
          boxHeight: 8,
          color: '#475569',
          font: chartFont
        }
      },
      tooltip: {
        ...commonTooltip,
        callbacks: {
          label: (context: any) => ` ${context.label}: ${context.raw}%`
        }
      }
    },
    cutout: '75%'
  };

  return <Doughnut data={data} options={options} />;
};


// --- 4. MONTHLY SPENT CHART ---
interface MonthlySpentChartProps {
  totalSpent: number;
}
export const MonthlySpentChart: React.FC<MonthlySpentChartProps> = ({ totalSpent }) => {
  const currentMonthIdx = new Date().getMonth();
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  
  const monthlyData = months.map((_, i) => {
    if (i === currentMonthIdx) return totalSpent;
    return 0;
  });

  const data = {
    labels: months,
    datasets: [{
      label: 'Spending',
      data: monthlyData,
      backgroundColor: '#4f46e5',
      borderRadius: 4,
      barPercentage: 0.6
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: commonTooltip
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: {
          ...commonTicks,
          callback: (value: any) => '₹' + value.toLocaleString()
        }
      }
    }
  };

  return <Bar data={data} options={options} />;
};


// --- 5. CATEGORY CHART ---
interface CategoryChartProps {
  byCategory: Record<string, number>;
}
export const CategoryChart: React.FC<CategoryChartProps> = ({ byCategory }) => {
  const categories = ['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Savings'];
  const chartData = categories.map(cat => byCategory[cat.toLowerCase()] || byCategory[cat] || 0);

  const data = {
    labels: categories,
    datasets: [{
      data: chartData,
      backgroundColor: [
        '#4f46e5', // Indigo
        '#06b6d4', // Cyan
        '#14b8a6', // Teal
        '#f59e0b', // Amber
        '#f43f5e', // Rose
        '#8b5cf6'  // Violet
      ],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: {
          boxWidth: 8,
          boxHeight: 8,
          color: '#475569',
          font: chartFont
        }
      },
      tooltip: {
        ...commonTooltip,
        callbacks: {
          label: (context: any) => ` ${context.label}: ₹${context.raw.toLocaleString()}`
        }
      }
    },
    cutout: '75%'
  };

  return <Doughnut data={data} options={options} />;
};


// --- 6. BUDGET VS SPENT CHART ---
interface BudgetVsSpentChartProps {
  budgetSplit: Record<string, number> | null;
  byCategory: Record<string, number>;
}
export const BudgetVsSpentChart: React.FC<BudgetVsSpentChartProps> = ({ budgetSplit, byCategory }) => {
  const categories = ['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Savings'];
  
  const budgetData = categories.map(cat => budgetSplit?.[cat] || budgetSplit?.[cat.toLowerCase()] || 0);
  const spentData = categories.map(cat => byCategory[cat.toLowerCase()] || byCategory[cat] || 0);

  const data = {
    labels: categories,
    datasets: [
      {
        label: 'Budget',
        data: budgetData,
        backgroundColor: 'rgba(79, 70, 229, 0.7)',
        borderColor: '#4f46e5',
        borderWidth: 1,
        borderRadius: 4,
        barPercentage: 0.6
      },
      {
        label: 'Spent',
        data: spentData,
        backgroundColor: 'rgba(244, 63, 94, 0.7)',
        borderColor: '#f43f5e',
        borderWidth: 1,
        borderRadius: 4,
        barPercentage: 0.6
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 8,
          boxHeight: 8,
          color: '#475569',
          font: chartFont
        }
      },
      tooltip: commonTooltip
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: {
          ...commonTicks,
          callback: (value: any) => '₹' + value.toLocaleString()
        }
      }
    }
  };

  return <Bar data={data} options={options} />;
};


// --- 7. FORECAST CHART ---
interface ForecastChartProps {
  forecastData: {
    history?: Array<{ month: string; amount: number }>;
    forecast?: Array<{ month: string; predicted_amount: number }>;
    current_month_total?: number;
    next_month_forecast?: number;
  } | null;
}
export const ForecastChart: React.FC<ForecastChartProps> = ({ forecastData }) => {
  if (!forecastData || !forecastData.history) {
    return (
      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', opacity: 0.5, fontStyle: 'italic', fontSize: '13px' }}>
        Add more expenses to see forecasts
      </div>
    );
  }

  const historyLabels = forecastData.history.map(h => h.month);
  const historyValues = forecastData.history.map(h => h.amount);
  
  const forecastLabels = forecastData.forecast?.map(f => f.month) || [];
  const forecastValues = forecastData.forecast?.map(f => f.predicted_amount) || [];

  const combinedLabels = [...historyLabels, ...forecastLabels];
  const combinedHistory = [...historyValues, ...Array(forecastLabels.length).fill(null)];
  const combinedForecast = [...Array(historyLabels.length).fill(null), ...forecastValues];

  const data = {
    labels: combinedLabels,
    datasets: [
      {
        label: 'Historical Spending',
        data: combinedHistory,
        borderColor: '#4f46e5',
        backgroundColor: 'rgba(79, 70, 229, 0.04)',
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 1.5
      },
      {
        label: 'Forecasted Spending',
        data: combinedForecast,
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.04)',
        borderDash: [5, 5],
        tension: 0.4,
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 1.5
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Expense Forecasting (Linear Regression)',
        font: { ...chartFont, weight: 'bold' as const, size: 12 },
        color: '#0f172a'
      },
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 8,
          boxHeight: 8,
          color: '#475569',
          font: chartFont
        }
      },
      tooltip: commonTooltip
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: {
          ...commonTicks,
          callback: (value: any) => '₹' + value.toLocaleString()
        }
      }
    }
  };

  return <Line data={data} options={options} />;
};


// --- 8. STOCK PRICE CHART ---
interface StockPriceChartProps {
  stockData: Array<{ Date: string; Close: number }> | null;
  symbol: string;
}
export const StockPriceChart: React.FC<StockPriceChartProps> = ({ stockData, symbol }) => {
  if (!stockData || stockData.length === 0) {
    return (
      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', opacity: 0.5, fontStyle: 'italic', fontSize: '13px' }}>
        No historical data available for {symbol}
      </div>
    );
  }

  const labels = stockData.map(d => {
    try {
      const dateObj = new Date(d.Date);
      return dateObj.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
    } catch {
      return d.Date;
    }
  });
  const values = stockData.map(d => d.Close);

  const data = {
    labels,
    datasets: [
      {
        label: `${symbol} Close Price`,
        data: values,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.04)',
        tension: 0.2,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        ...commonTooltip,
        callbacks: {
          label: (context: any) => ` Price: ₹${Number(context.raw).toFixed(2)}`
        }
      }
    },
    scales: {
      x: {
        grid: commonGridX,
        ticks: commonTicks
      },
      y: {
        grid: commonGridY,
        ticks: {
          ...commonTicks,
          callback: (value: any) => '₹' + Number(value).toLocaleString('en-IN')
        }
      }
    }
  };

  return <Line data={data} options={options} />;
};

