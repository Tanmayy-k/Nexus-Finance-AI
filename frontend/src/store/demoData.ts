export const demoData = {
  summary: {
    total_spent: 45000,
    by_category: { 'Housing': 20000, 'Food': 15000, 'Transportation': 5000, 'Entertainment': 5000 },
    alerts: []
  },
  expenses: [
    { id: 1, category: 'Food', amount: 1500, description: 'Lunch out', date: '2026-06-16' },
    { id: 2, category: 'Housing', amount: 20000, description: 'Rent', date: '2026-06-01' },
    { id: 3, category: 'Transportation', amount: 500, description: 'Uber', date: '2026-06-15' },
    { id: 4, category: 'Entertainment', amount: 5000, description: 'Concert', date: '2026-06-10' }
  ],
  anomalies: {
    has_data: true,
    total_analyzed: 50,
    anomalies: [{ id: 1, category: 'Entertainment', amount: 5000, date: '2026-06-10', description: 'Concert Tickets', anomaly_score: 0.95, reason: 'Unusually high amount for Entertainment' }],
    message: '1 anomaly found'
  },
  forecast: {
    has_data: true,
    months_available: 3,
    current_month_total: 45000,
    next_month_forecast: 42000,
    change_pct: -6.6,
    trend: 'decreasing',
    message: 'Spending is trending down',
    history: [{ month: 'Apr', amount: 50000 }, { month: 'May', amount: 48000 }, { month: 'Jun', amount: 45000 }],
    forecast: [{ month: 'Jul', predicted_amount: 42000 }]
  },
  insights: {
    insights: [
      { type: 'spending', severity: 'warning', message: 'You spent 30% more on Food this month.' }, 
      { type: 'savings', severity: 'positive', message: 'You are on track to meet your savings goal.' },
      { type: 'tips', severity: 'info', message: 'Consider switching to a high-yield savings account.' }
    ],
    summary: 'Overall, your spending is slightly high but manageable.'
  },
  weekly_focus: { focus_message: 'Focus on reducing Food expenses this week.' },
  recommendations: {
    recommendations: [
      { scheme: 'Index Fund A', reason: 'Low risk, steady growth' },
      { scheme: 'Tech ETF', reason: 'High growth potential' }
    ]
  },
  portfolio_valuation: {
    net_worth: 2487590,
    cash_available: 312450,
    total_invested: 2175140,
    current_value: 2350000,
    absolute_gain: 174860,
    gain_percentage: 8.04
  },
  budget: {
    has_budget: true,
    income: 80000,
    goal: 'Retirement',
    budget_split: { 'Housing': 20000, 'Food': 15000, 'Transportation': 5000, 'Entertainment': 10000, 'Savings': 30000 }
  },
  tips: { tips: ['Cook at home to save on Food.', 'Consider a cheaper gym membership.'] },
  cluster_users: {
    current_user_cluster: 1, 
    clusters: [
      { id: 1, income: 80000, goal: 1, risk: 2, cluster: 1 },
      { id: 2, income: 75000, goal: 1, risk: 2, cluster: 1 },
      { id: 3, income: 50000, goal: 0, risk: 0, cluster: 0 },
      { id: 4, income: 120000, goal: 2, risk: 1, cluster: 2 }
    ]
  },
  stocks_list: { available_symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'] },
  stock_data: [
    { Date: '2026-06-01', Close: 150, Open: 148, High: 152, Low: 147, Volume: 10000, Symbol: 'AAPL' }, 
    { Date: '2026-06-02', Close: 155, Open: 151, High: 156, Low: 150, Volume: 12000, Symbol: 'AAPL' },
    { Date: '2026-06-05', Close: 160, Open: 156, High: 162, Low: 155, Volume: 15000, Symbol: 'AAPL' },
    { Date: '2026-06-10', Close: 158, Open: 161, High: 161, Low: 157, Volume: 11000, Symbol: 'AAPL' },
    { Date: '2026-06-15', Close: 165, Open: 159, High: 166, Low: 158, Volume: 14000, Symbol: 'AAPL' }
  ],
  predict: {
    prediction: 82,
    score: 82,
    factors: {
      positive: [{ label: 'High savings rate', impact: 10 }],
      negative: [{ label: 'High food spend', impact: -2 }]
    }
  },
  copilot_chat: {
    reply: 'Based on your data, you should cut back on entertainment. You have already spent ₹5000 on concert tickets this month, putting you near your limit.'
  }
};
