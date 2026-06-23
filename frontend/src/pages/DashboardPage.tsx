import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { DemoTour } from '../components/DemoTour';
import { useAuthStore } from '../store/useAuthStore';
import { useBudgetStore } from '../store/useBudgetStore';
import {
  useExpenses,
  useFinancialSummary,
  useAnomalies,
  useForecast,
  useInsights,
  useWeeklyFocus,
  useDeleteExpense,
  useBudgetTips
} from '../hooks/useExpenses';
import {
  useRecommendations,
  useUserClustering,
  usePortfolioValuation
} from '../hooks/useRecommendations';
import {
  useCopilotChat,
  usePredictScore
} from '../hooks/useCopilot';
import {
  useStocksList,
  useStockData
} from '../hooks/useStocks';

import {
  AllocationChart,
  MonthlySpentChart,
  CategoryChart,
  ForecastChart,
  StockPriceChart
} from '../components/Charts/Charts';


import { ExpenseModal } from '../components/Modals/ExpenseModal';
import { BudgetModal } from '../components/Modals/BudgetModal';
import { RiskQuizModal } from '../components/Modals/RiskQuizModal';

import '../styles/dashboard.css';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout, setProfile } = useAuthStore();
  const { budgetSplit, fetchBudget } = useBudgetStore();

  // Tab & UI Layout State
  const [activeSection, setActiveSection] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Modals state
  const [expenseModalOpen, setExpenseModalOpen] = useState(false);
  const [budgetModalOpen, setBudgetModalOpen] = useState(false);
  const [riskQuizModalOpen, setRiskQuizModalOpen] = useState(false);

  // Investment Section selections
  const [investGoal, setInvestGoal] = useState('Growth');
  const [investRisk, setInvestRisk] = useState('Medium');
  const [investHorizon, setInvestHorizon] = useState('Medium');
  const [investSector, setInvestSector] = useState('Any');

  // Filter & Sort recommendations
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortBy, setSortBy] = useState('default');

  // Profile fields bindings for settings
  const [profileName, setProfileName] = useState(user?.name || '');
  const [profileEmail, setProfileEmail] = useState(user?.email || '');
  const [profileIncome, setProfileIncome] = useState<number | string>(user?.income || '');
  const [profileGoal, setProfileGoal] = useState(user?.goal || 'Growth');
  const [profileRisk, setProfileRisk] = useState(user?.risk_profile || 'medium');
  const [appTheme, setAppTheme] = useState(localStorage.getItem('nexus_theme') || 'light');

  // Copilot messages state
  const [copilotMessages, setCopilotMessages] = useState<Array<{ text: string; sender: 'user' | 'bot' | 'error' }>>([
    { text: "👋 Hi! I'm your AI financial advisor. Ask me anything about your finances.", sender: 'bot' }
  ]);
  const [copilotInput, setCopilotInput] = useState('');
  


  // Floating AI state
  const [floatingAiOpen, setFloatingAiOpen] = useState(false);
  const [floatingAiLoading, setFloatingAiLoading] = useState(false);
  const [floatingAiMessages, setFloatingAiMessages] = useState<Array<{ text: string; sender: 'user' | 'bot'; isError?: boolean; originalText?: string }>>([
    { text: "Hi! I am **Nexus AI**, your personal financial assistant. I'm aware of what you are viewing on the screen. How can I help you today?", sender: 'bot' }
  ]);
  const [floatingAiInput, setFloatingAiInput] = useState('');

  // Scroll references
  const copilotHistoryRef = useRef<HTMLDivElement>(null);
  const floatingHistoryRef = useRef<HTMLDivElement>(null);

  // Financial advice score state
  const [financialScore, setFinancialScore] = useState<number | null>(null);
  const [scoreFactors, setScoreFactors] = useState<any>(null);

  // Queries
  const { data: expenses = [], refetch: refetchExpenses } = useExpenses();
  const { data: summary, refetch: refetchSummary } = useFinancialSummary();
  const { data: anomalies, refetch: refetchAnomalies } = useAnomalies();
  const { data: forecast, refetch: refetchForecast } = useForecast();
  const { data: insights, refetch: refetchInsights } = useInsights();
  const { refetch: refetchWeeklyFocus } = useWeeklyFocus();
  const { data: recommendations, refetch: refetchRecs } = useRecommendations();
  const { data: clustering, refetch: refetchClustering } = useUserClustering();
  const { data: portfolioValuation, refetch: refetchValuation } = usePortfolioValuation();
  const { data: budgetTipData, isLoading: tipsLoading, refetch: refetchBudgetTips } = useBudgetTips();

  // Stock market states & queries
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const { data: stockSymbols = [] } = useStocksList();
  const { data: stockHistory = [], isLoading: stockLoading } = useStockData(selectedStock);

  // Mutations
  const copilotChatMutation = useCopilotChat();
  const predictScoreMutation = usePredictScore();
  const deleteExpenseMutation = useDeleteExpense();


  // Transactions local table states
  const [txSearchQuery, setTxSearchQuery] = useState('');
  const [txCategoryFilter, setTxCategoryFilter] = useState('all');
  const [txSelectedIds, setTxSelectedIds] = useState<Array<number | string>>([]);
  const [txCurrentPage, setTxCurrentPage] = useState(1);
  const txItemsPerPage = 8;

  // Sync profile fields when user loads
  useEffect(() => {
    if (user) {
      setProfileName(user.name || '');
      setProfileEmail(user.email || '');
      setProfileIncome(user.income || '');
      setProfileGoal(user.goal || 'Growth');
    }
  }, [user]);

  // Fetch initial budget on load
  useEffect(() => {
    fetchBudget();
  }, [fetchBudget]);

  // Scroll to bottom on new messages
  useEffect(() => {
    if (copilotHistoryRef.current) {
      copilotHistoryRef.current.scrollTop = copilotHistoryRef.current.scrollHeight;
    }
  }, [copilotMessages]);



  useEffect(() => {
    if (floatingHistoryRef.current) {
      floatingHistoryRef.current.scrollTop = floatingHistoryRef.current.scrollHeight;
    }
  }, [floatingAiMessages]);

  // Apply theme changes
  useEffect(() => {
    if (appTheme === 'light') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', appTheme);
    }
    localStorage.setItem('nexus_theme', appTheme);
  }, [appTheme]);

  // Compute financial health score whenever summary data updates
  useEffect(() => {
    if (summary) {
      if (!user?.income) {
        setFinancialScore(null);
        setScoreFactors(null);
        return;
      }
      
      const income = user.income;
      const spent = summary.total_spent;
      const investmentAmt = income * 0.1;

      predictScoreMutation.mutate(
        {
          monthly_income: income,
          monthly_expense_total: spent,
          investment_amount: investmentAmt
        },
        {
          onSuccess: (data) => {
            setFinancialScore(data.score ?? data.prediction);
            setScoreFactors(data.factors);
          },
          onError: (err) => {
            console.error("Score prediction failed:", err);
          }
        }
      );
    }
  }, [summary, user]);

  // Invalidate and refetch all data
  const handleRefreshData = () => {
    refetchExpenses();
    refetchSummary();
    refetchAnomalies();
    refetchForecast();
    refetchInsights();
    refetchWeeklyFocus();
    refetchRecs();
    refetchClustering();
    fetchBudget();
    refetchValuation();
    refetchBudgetTips();
    (window as any).showToast('Data refreshed', 'success');
  };

  const handleExport = () => {
    (window as any).showToast('Export started. You will receive an email shortly.', 'success');
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  // Profile save settings
  const handleSaveProfile = () => {
    setProfile({
      name: profileName,
      email: profileEmail,
      income: Number(profileIncome),
      goal: profileGoal
    });
    (window as any).showSuccess('Settings saved successfully!');
  };

  // Chat queries submit handlers
  const handleSendCopilotMessage = async (textToSend?: string) => {
    const text = (textToSend || copilotInput).trim();
    if (!text) return;

    setCopilotInput('');
    setCopilotMessages(prev => [...prev, { text, sender: 'user' }]);
    
    // Append loading text
    setCopilotMessages(prev => [...prev, { text: 'Thinking...', sender: 'bot' }]);

    try {
      const response = await copilotChatMutation.mutateAsync({ message: text, current_page: 'analytics' });
      // Remove last 'Thinking...' item and append reply
      setCopilotMessages(prev => {
        const copy = [...prev];
        copy.pop(); // remove 'Thinking...'
        return [...copy, { text: response.reply, sender: 'bot' }];
      });
    } catch (error) {
      setCopilotMessages(prev => {
        const copy = [...prev];
        copy.pop(); // remove 'Thinking...'
        return [...copy, { text: 'Connection error. Check your server settings.', sender: 'error' }];
      });
    }
  };

  // Helper for rendering Markdown in chat bubbles
  const formatMessageText = (text: string) => {
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      let formatted = line;
      // Bold: **text**
      formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      // Italic: *text*
      formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
      
      // Check for bullet items
      if (formatted.trim().startsWith('- ') || formatted.trim().startsWith('* ')) {
        const cleanContent = formatted.replace(/^[-*]\s+/, '');
        return (
          <li key={idx} style={{ marginLeft: '16px', listStyleType: 'disc', margin: '4px 0' }} dangerouslySetInnerHTML={{ __html: cleanContent }} />
        );
      }
      
      return (
        <p key={idx} style={{ margin: '0 0 6px 0', minHeight: '1.2em' }} dangerouslySetInnerHTML={{ __html: formatted }} />
      );
    });
  };

  // Helper for generating context-aware suggested prompts
  const getSuggestedPrompts = (page: string) => {
    switch (page) {
      case 'dashboard':
        return [
          "Explain my net worth details",
          "Summarize my monthly vitals",
          "What are my key next actions?"
        ];
      case 'analytics':
        return [
          "What factors drive my health score?",
          "Analyze my utilities spending",
          "Explain my linear regression forecast"
        ];
      case 'investments':
        return [
          "Recommend mutual funds for high risk",
          "Analyze my current portfolio allocation",
          "Should I buy tech or bond assets?"
        ];
      case 'budget':
        return [
          "Check my category budget utilization",
          "Why was my shopping category flagged?",
          "Give me custom budget saving tips"
        ];
      case 'transactions':
        return [
          "Search my grocery expenses",
          "Find any anomalous charges",
          "Show spending by category"
        ];
      case 'settings':
        return [
          "How do I update my monthly income?",
          "What risk preference matches growth?",
          "Help me configure a savings goal"
        ];
      default:
        return [
          "Analyze my spending habits",
          "Recommend savings opportunities",
          "How can I improve my financial score?"
        ];
    }
  };

  const handleSendFloatingAiMessage = async (textToSend?: string) => {
    const text = (textToSend !== undefined ? textToSend : floatingAiInput).trim();
    if (!text) return;

    if (textToSend === undefined) {
      setFloatingAiInput('');
    }

    // Add user message to UI
    setFloatingAiMessages(prev => [...prev, { text, sender: 'user' }]);
    setFloatingAiLoading(true);

    try {
      const response = await copilotChatMutation.mutateAsync({ message: text, current_page: activeSection });
      setFloatingAiMessages(prev => [...prev, { text: response.reply, sender: 'bot' }]);
    } catch (error) {
      setFloatingAiMessages(prev => [
        ...prev, 
        { 
          text: 'Connection error. Check your server settings.', 
          sender: 'bot', 
          isError: true, 
          originalText: text 
        }
      ]);
    } finally {
      setFloatingAiLoading(false);
    }
  };

  // Quiz completed callback
  const handleQuizComplete = (level: 'Low' | 'Medium' | 'High', score: number) => {
    setInvestRisk(level);
    (window as any).showToast(`Risk profile computed: ${level} (${score}/100)`, 'success');
  };

  // UI calculations
  const totalSpent = summary?.total_spent || 0;
  const categoriesBreakdown = summary?.by_category || {};

  const topCategoryData = Object.keys(categoriesBreakdown).reduce<{ name: string; amount: number }>(
    (acc, cat) => {
      const amt = categoriesBreakdown[cat];
      return amt > acc.amount ? { name: cat, amount: amt } : acc;
    },
    { name: 'None', amount: 0 }
  );

  // Budget calculations
  const totalBudgetLimit = budgetSplit
    ? Object.values(budgetSplit).reduce((a, b) => a + b, 0)
    : 0;

  // Sorting and filtering recommendations
  const getFilteredRecs = () => {
    if (!recommendations?.recommendations) return [];
    
    let list = [...recommendations.recommendations];

    // Filter
    if (riskFilter !== 'all') {
      list = list.filter(r => {
        const lower = r.scheme.toLowerCase();
        if (riskFilter === 'low') return lower.includes('bond') || lower.includes('debt') || lower.includes('fixed') || lower.includes('deposit');
        if (riskFilter === 'high') return lower.includes('equity') || lower.includes('stock') || lower.includes('crypto');
        return !lower.includes('bond') && !lower.includes('debt') && !lower.includes('equity') && !lower.includes('stock');
      });
    }

    // Sort
    if (sortBy === 'risk') {
      list.sort((a, b) => {
        const getRisk = (name: string) => {
          const lower = name.toLowerCase();
          if (lower.includes('equity') || lower.includes('stock')) return 3;
          if (lower.includes('bond') || lower.includes('debt') || lower.includes('fixed')) return 1;
          return 2;
        };
        return getRisk(a.scheme) - getRisk(b.scheme);
      });
    } else if (sortBy === 'investment') {
      list.sort((a, b) => {
        const getMin = (name: string) => {
          const lower = name.toLowerCase();
          if (lower.includes('sip') || lower.includes('mutual')) return 500;
          if (lower.includes('fixed') || lower.includes('fd')) return 1000;
          return 5000;
        };
        return getMin(a.scheme) - getMin(b.scheme);
      });
    } else if (sortBy === 'name') {
      list.sort((a, b) => a.scheme.localeCompare(b.scheme));
    }

    return list;
  };

  // Get filtered transactions for search & pagination
  const getFilteredTransactions = () => {
    let list = [...expenses];
    if (txSearchQuery.trim() !== '') {
      const q = txSearchQuery.toLowerCase();
      list = list.filter(t => 
        (t.description && t.description.toLowerCase().includes(q)) || 
        (t.category && t.category.toLowerCase().includes(q))
      );
    }
    if (txCategoryFilter !== 'all') {
      list = list.filter(t => t.category.toLowerCase() === txCategoryFilter.toLowerCase());
    }
    return list;
  };

  // Transactions list variables
  const filteredTxs = getFilteredTransactions();
  const totalTxPages = Math.max(1, Math.ceil(filteredTxs.length / txItemsPerPage));
  const txStartIndex = (txCurrentPage - 1) * txItemsPerPage;
  const paginatedTxs = filteredTxs.slice(txStartIndex, txStartIndex + txItemsPerPage);

  const toggleSelectTx = (id: number | string) => {
    setTxSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const toggleSelectAllTxs = (pageTxs: any) => {
    const pageIds: Array<number | string> = (pageTxs || []).map((t: any) => t.id);
    const allSelected = pageIds.length > 0 && pageIds.every(id => txSelectedIds.includes(id));
    if (allSelected) {
      setTxSelectedIds(prev => prev.filter(id => !pageIds.includes(id)));
    } else {
      setTxSelectedIds(prev => [...new Set([...prev, ...pageIds])]);
    }
  };

  const handleBulkDelete = async () => {
    if (txSelectedIds.length === 0) return;
    if (window.confirm(`Are you sure you want to delete ${txSelectedIds.length} transaction(s)?`)) {
      try {
        await Promise.all(txSelectedIds.map(id => deleteExpenseMutation.mutateAsync(id)));
        (window as any).showSuccess(`${txSelectedIds.length} transaction(s) deleted successfully`);
        setTxSelectedIds([]);
      } catch (err) {
        console.error("Failed to delete transactions:", err);
        (window as any).showToast('Error deleting transactions', 'error');
      }
    }
  };

  // Get user avatar initials
  const getAvatarInitials = () => {
    if (!user?.name) return 'U';
    return user.name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
  };

  return (
    <div className="dashboard-body">
      <button 
        className="mobile-menu-btn" 
        id="mobile-menu-btn"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        <span className="material-icons">menu</span>
      </button>

      <button 
        className={`desktop-menu-btn ${sidebarCollapsed ? 'show' : ''}`} 
        onClick={() => setSidebarCollapsed(false)}
        title="Open Sidebar"
      >
        <span className="material-icons">menu</span>
      </button>

      {/* Mobile overlay */}
      <div 
        className={`mobile-sidebar-overlay ${sidebarOpen ? 'active' : ''}`} 
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      ></div>

      <div className="app-container">
        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : ''} ${sidebarCollapsed ? 'collapsed' : ''}`} id="sidebar">
          <div className="logo" style={{ cursor: 'pointer', justifyContent: sidebarCollapsed ? 'center' : 'flex-start' }} onClick={() => setSidebarCollapsed(!sidebarCollapsed)} title="Toggle Sidebar">
            <div className="logo-icon">
              <span className="material-icons">auto_awesome</span>
            </div>
            {!sidebarCollapsed && <div className="logo-text">Nexus Finance</div>}
          </div>
          
          <nav className="nav-menu" role="navigation" aria-label="Main Navigation">
            <a 
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'dashboard' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('dashboard'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('dashboard'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">dashboard</span>
              <span className="nav-item-label">Dashboard</span>
            </a>
            <a 
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'analytics' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('analytics'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('analytics'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">show_chart</span>
              <span className="nav-item-label">Analytics</span>
            </a>
            <a 
              id="nav-investments"
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'investments' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('investments'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('investments'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">trending_up</span>
              <span className="nav-item-label">Investments</span>
            </a>
            <a 
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'transactions' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('transactions'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('transactions'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">receipt</span>
              <span className="nav-item-label">Transactions</span>
            </a>
            <a 
              id="nav-budget"
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'budget' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('budget'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('budget'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">savings</span>
              <span className="nav-item-label">Budget</span>
            </a>
            <a 
              role="button"
              tabIndex={0}
              className={`nav-item ${activeSection === 'settings' ? 'active' : ''}`} 
              onClick={() => { setActiveSection('settings'); setSidebarOpen(false); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setActiveSection('settings'); setSidebarOpen(false); } }}
            >
              <span className="material-icons">settings</span>
              <span className="nav-item-label">Settings</span>
            </a>
          </nav>
          
          <div className="user-profile" id="user-profile">
            <div className="user-avatar">{getAvatarInitials()}</div>
            <div className="user-info">
              <div className="user-name">{user?.name || 'User'}</div>
              <div className="user-email">{user?.email || ''}</div>
            </div>
            <button className="card-action-btn" onClick={handleLogout} aria-label="Log out of application">
              <span className="material-icons">logout</span>
            </button>
          </div>
        </aside>
        
        {/* Main Content */}
        <main className={`main-content ${sidebarCollapsed ? 'expanded' : ''}`}>
          {/* Header */}
          <header className="dashboard-header">
            <h1 className="header-title">
              {activeSection === 'dashboard' && 'Financial Dashboard'}
              {activeSection === 'analytics' && 'Detailed Analytics'}
              {activeSection === 'investments' && 'Investment Planner'}
              {activeSection === 'transactions' && 'Transaction History'}
              {activeSection === 'budget' && 'Budget Planner'}
              {activeSection === 'settings' && 'Account Settings'}
            </h1>
            <div className="header-actions">

              <button 
                className="header-btn primary" 
                id="new-transaction-btn" 
                onClick={() => setExpenseModalOpen(true)}
              >
                <span className="material-icons">add</span>
                New Transaction
              </button>
            </div>
          </header>
          
          {/* 1. Dashboard Tab */}
          {activeSection === 'dashboard' && (
            <section className="content-section" id="section-dashboard">
              {/* Welcome header */}
              <div className="welcome-header" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-heading)' }}>
                  Welcome back, {user?.name || 'User'}!
                </h2>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  Here is an overview of your financial status.
                </p>
              </div>

              <div className="overview-metrics" style={{ borderBottom: 'none' }}>
                <div className="metric-card">
                  <div className="metric-label">
                    <span className="material-icons" style={{ color: 'var(--primary)' }}>account_balance</span>
                    Net Worth
                  </div>
                  <div className="metric-value">
                    ₹{portfolioValuation ? Math.round(portfolioValuation.net_worth).toLocaleString('en-IN') : '...'}
                  </div>
                  <div className="metric-change positive">
                    <span className="material-icons">trending_up</span>
                    4.2% this month
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-label">
                    <span className="material-icons" style={{ color: 'var(--success)' }}>savings</span>
                    Cash Available
                  </div>
                  <div className="metric-value">
                    ₹{portfolioValuation ? Math.round(portfolioValuation.cash_available).toLocaleString('en-IN') : '...'}
                  </div>
                  <div className="metric-change positive">
                    <span className="material-icons">trending_up</span>
                    2.1% this month
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-label">
                    <span className="material-icons" style={{ color: 'var(--danger)' }}>credit_card</span>
                    Monthly Spending
                  </div>
                  <div className="metric-value">₹{totalSpent.toLocaleString('en-IN')}</div>
                  <div className="metric-change negative">
                    <span className="material-icons">trending_down</span>
                    1.3% this month
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-label">
                    <span className="material-icons" style={{ color: 'var(--warning)' }}>show_chart</span>
                    Investment Return
                  </div>
                  <div className="metric-value">
                    {portfolioValuation ? portfolioValuation.gain_percentage : '...'}%
                  </div>
                  <div className="metric-change positive">
                    <span className="material-icons">trending_up</span>
                    +₹{portfolioValuation ? Math.round(portfolioValuation.absolute_gain).toLocaleString('en-IN') : '...'} return
                  </div>
                </div>
              </div>

              <div className="dashboard-grid">
                {/* Spending Trend Chart */}
                <section className="card" style={{ gridColumn: 'span 8' }}>
                  <div className="card-header">
                    <h2 className="card-title">Spending Trend</h2>
                    <div className="card-actions">
                      <select id="networth-range" defaultValue="12">
                        <option value="3">Last 3M</option>
                        <option value="6">Last 6M</option>
                        <option value="12">Last 12M</option>
                      </select>
                    </div>
                  </div>
                  <div className="card-body" style={{ height: '280px' }}>
                    <MonthlySpentChart totalSpent={totalSpent} />
                  </div>
                </section>

                {/* Portfolio Asset Allocation Chart */}
                <section className="card portfolio-allocation" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <h2 className="card-title">Asset Allocation</h2>
                  </div>
                  <div className="card-body" style={{ height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '10px' }}>
                    <AllocationChart riskProfile={user?.risk_profile || 'medium'} />
                  </div>
                </section>

                {/* Recent Transactions List */}
                <section className="card" style={{ gridColumn: 'span 12' }}>
                  <div className="card-header">
                    <h2 className="card-title">Recent Transactions</h2>
                    <button className="header-btn" onClick={() => setActiveSection('transactions')} style={{ fontSize: '11px', padding: '4px 8px' }}>
                      View All
                    </button>
                  </div>
                  <div className="card-body" style={{ padding: 0 }}>
                    <div className="transactions-list">
                      {expenses.length > 0 ? (
                        expenses.slice(0, 5).map((exp, i) => (
                          <div key={exp.id || i} className="transaction-row" style={{ gridTemplateColumns: '100px 1fr 100px', borderBottom: '1px solid var(--border)' }}>
                            <div className="t-date" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{exp.date}</div>
                            <div className="t-desc" style={{ fontSize: '13px', fontWeight: 600 }}>{exp.description || exp.category}</div>
                            <div className="t-amt" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>₹{Number(exp.amount).toFixed(2)}</div>
                          </div>
                        ))
                      ) : (
                        <div style={{ padding: '30px 20px', textAlign: 'center', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', margin: '10px 0' }}>
                          <span className="material-icons" style={{ fontSize: '32px', color: 'var(--text-muted)', marginBottom: '10px', display: 'block' }}>receipt_long</span>
                          <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>No transactions yet</h4>
                          <p style={{ opacity: 0.7, fontSize: '12px', lineHeight: 1.5, marginBottom: '15px' }}>
                            Add your first expense to begin understanding your spending habits and unlock AI insights.
                          </p>
                          <button className="header-btn primary" onClick={() => setExpenseModalOpen(true)} style={{ margin: '0 auto', fontSize: '12px', padding: '6px 12px' }}>
                            Add First Expense
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </section>

                {/* Bottom Row - Goals, Budget health, Upcoming bills, Recommendations */}
                {/* 1. Goals progress */}
                <section className="card" style={{ gridColumn: 'span 3' }}>
                  <div className="card-header">
                    <h2 className="card-title">Savings Goals</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '8px', justifyContent: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600 }}>
                      <span>Retirement Goal</span>
                      <span>65%</span>
                    </div>
                    <div className="budget-progress" style={{ margin: 0 }}>
                      <div className="budget-progress-bar" style={{ width: '65%' }}></div>
                    </div>
                    <p style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      On track to hit ₹10L by 2028.
                    </p>
                  </div>
                </section>

                {/* 2. Budget Health */}
                <section className="card" style={{ gridColumn: 'span 3' }}>
                  <div className="card-header">
                    <h2 className="card-title">Budget Health</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                      <span>Spent: ₹{totalSpent.toLocaleString('en-IN')}</span>
                      <span>Limit: ₹{totalBudgetLimit.toLocaleString('en-IN')}</span>
                    </div>
                    {budgetSplit ? (
                      (['Housing', 'Food', 'Entertainment'] as const).map((cat) => {
                        const limit = budgetSplit[cat] || 0;
                        const spent = categoriesBreakdown[cat.toLowerCase()] || categoriesBreakdown[cat] || 0;
                        const pct = limit > 0 ? (spent / limit) * 100 : 0;
                        
                        let barColor = 'var(--primary)';
                        if (pct >= 100) barColor = 'var(--danger)';
                        else if (pct >= 80) barColor = 'var(--warning, #f59e0b)';

                        return (
                          <div key={cat} style={{ marginBottom: '4px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
                              <span style={{ fontWeight: 600 }}>{cat}</span>
                              <span style={{ color: pct >= 100 ? 'var(--danger)' : 'var(--text-primary)' }}>
                                {pct >= 100 ? 'OVER LIMIT' : `₹${(limit - spent).toLocaleString('en-IN')} left`}
                              </span>
                            </div>
                            <div className="budget-progress" style={{ margin: 0, height: '6px' }}>
                              <div style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: barColor, height: '100%', borderRadius: '4px' }}></div>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <p style={{ fontSize: '11px', opacity: 0.6, textAlign: 'center' }}>Configure your budget envelopes to see them here.</p>
                    )}
                  </div>
                </section>

                {/* 3. Upcoming Bills */}
                <section className="card" style={{ gridColumn: 'span 3' }}>
                  <div className="card-header">
                    <h2 className="card-title">Upcoming Bills</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '8px', justifyContent: 'center', fontSize: '11px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
                      <span>Rent (Landlord)</span>
                      <strong>₹15,000 (Jun 20)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
                      <span>Internet Bill</span>
                      <strong>₹850 (Jun 22)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Gym Fee</span>
                      <strong>₹2,000 (Jun 25)</strong>
                    </div>
                  </div>
                </section>

                {/* 4. AI Recommendations list */}
                <section className="card" style={{ gridColumn: 'span 3' }}>
                  <div className="card-header">
                    <h2 className="card-title">Curated Suggestions</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '6px', justifyContent: 'center', fontSize: '11px' }}>
                    {recommendations?.recommendations ? (
                      recommendations.recommendations.slice(0, 2).map((rec, i) => (
                        <div key={i} style={{ borderBottom: i === 0 ? '1px solid var(--border)' : 'none', paddingBottom: i === 0 ? '6px' : '0' }}>
                          <span style={{ fontWeight: 600, color: 'var(--primary)', display: 'block' }}>{rec.scheme}</span>
                          <span style={{ opacity: 0.7, fontSize: '10px', textOverflow: 'ellipsis', overflow: 'hidden', display: 'block', whiteSpace: 'nowrap' }}>{rec.reason}</span>
                        </div>
                      ))
                    ) : (
                      <span style={{ opacity: 0.6 }}>No fund recommendations yet.</span>
                    )}
                  </div>
                </section>
              </div>
            </section>
          )}

          {/* 2. Analytics Tab (Completely Overhauled Layout) */}
          {activeSection === 'analytics' && (
            <section className="content-section" id="section-analytics">
              {/* 1. Financial Health Score Hero */}
              <div className="card" style={{ width: '100%' }}>
                <div className="card-header">
                  <h2 className="card-title">Financial Health Analysis</h2>
                </div>
                <div className="card-body" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '30px', alignItems: 'center' }}>
                  {/* Left Hero Block */}
                  <div id="score-target" style={{ textAlign: 'center', borderRight: '1px solid var(--border)', paddingRight: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ position: 'relative', width: '120px', height: '120px', margin: '0 auto 12px' }}>
                      <svg width="120" height="120" className="health-score-svg" aria-label={`Financial Health Score: ${financialScore !== null ? financialScore : 'uncalculated'}`}>
                        <circle cx="60" cy="60" r="50" className="health-score-bg" />
                        <circle 
                          cx="60" 
                          cy="60" 
                          r="50" 
                          className="health-score-progress" 
                          strokeDasharray="314" 
                          strokeDashoffset={314 - (314 * (financialScore || 0)) / 100} 
                        />
                      </svg>
                      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                        <div style={{ fontSize: '36px', fontWeight: 700, fontFamily: 'var(--font-heading)', color: 'var(--primary)', lineHeight: 1 }}>
                          {financialScore !== null ? `${financialScore}` : '--'}
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Overall Health Score
                    </div>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>
                      {financialScore && financialScore >= 80 ? 'Excellent financial stance' : financialScore && financialScore >= 60 ? 'Moderate financial stance' : 'Needs attention'}
                    </p>
                  </div>
                  {/* Right Explainability Block (SHAP Factors) */}
                  <div id="shap-target">
                    <h3 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px' }}>Explainable AI (SHAP) Insights</h3>
                    {scoreFactors ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div>
                          <div style={{ color: 'var(--success)', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Positive Factors</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {(scoreFactors.positive || []).map((f: any, idx: number) => (
                              <span key={idx} className="chip low" style={{ fontSize: '11px' }}>
                                ✓ {f.label} ({f.impact > 0 ? `+${f.impact}` : f.impact})
                              </span>
                            ))}
                          </div>
                        </div>
                        <div style={{ marginTop: '5px' }}>
                          <div style={{ color: 'var(--danger)', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Negative Factors</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {(scoreFactors.negative || []).map((f: any, idx: number) => (
                              <span key={idx} className="chip high" style={{ fontSize: '11px' }}>
                                ✕ {f.label} ({f.impact})
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p style={{ opacity: 0.6, fontSize: '12px' }}>Computing SHAP details...</p>
                    )}
                  </div>
                </div>
              </div>

              {/* 2. Key Insights Summary Cards */}
              <div className="dashboard-grid">
                <div className="card" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <span className="card-title">⚠️ Unusual Activity</span>
                  </div>
                  <div className="card-body">
                    {anomalies ? (
                      anomalies.has_data ? (
                        anomalies.anomalies.length > 0 ? (
                          anomalies.anomalies.slice(0, 2).map((a, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: '12px' }}>
                              <div>
                                <span style={{ fontWeight: 600, display: 'block' }}>{a.category}</span>
                                <span style={{ opacity: 0.6, fontSize: '10px' }}>{a.reason}</span>
                              </div>
                              <strong style={{ color: 'var(--danger)' }}>₹{a.amount.toLocaleString()}</strong>
                            </div>
                          ))
                        ) : (
                          <p style={{ fontSize: '12px', opacity: 0.6, textAlign: 'center', padding: '10px 0' }}>✓ No unusual activity detected</p>
                        )
                      ) : (
                        <p style={{ fontSize: '12px', opacity: 0.6 }}>{anomalies.message}</p>
                      )
                    ) : (
                      <p style={{ fontSize: '12px', opacity: 0.6 }}>Analyzing spending...</p>
                    )}
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <span className="card-title">💡 Proactive Insights</span>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {insights ? (
                      insights.insights.length > 0 ? (
                        insights.insights.slice(0, 3).map((insight, i) => {
                          const icon = insight.severity === 'warning' ? '⚠️' : insight.severity === 'positive' ? '✅' : '💡';
                          return (
                            <div key={i} style={{ display: 'flex', gap: '8px', fontSize: '12px' }}>
                              <span>{icon}</span>
                              <span style={{ lineHeight: 1.3 }}>{insight.message}</span>
                            </div>
                          );
                        })
                      ) : (
                        <div style={{ textAlign: 'center', padding: '20px 0', opacity: 0.7 }}>
                          <span className="material-icons" style={{ fontSize: '24px', marginBottom: '8px', display: 'block' }}>auto_awesome</span>
                          <p style={{ fontSize: '12px', lineHeight: 1.5 }}>Your AI financial insights will appear here as Nexus learns about your habits.</p>
                        </div>
                      )
                    ) : (
                      <p style={{ fontSize: '12px', opacity: 0.6 }}>Generating insights...</p>
                    )}
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <span className="card-title">📊 Stats Breakdown</span>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
                      <span>Total spent this month</span>
                      <strong>₹{totalSpent.toLocaleString()}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
                      <span>Top Category</span>
                      <strong>{topCategoryData.name} (₹{topCategoryData.amount.toLocaleString()})</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Total logged transactions</span>
                      <strong>{expenses.length}</strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3 & 4. Spending Trends & Category Breakdown */}
              <div className="dashboard-grid">
                <div className="card" style={{ gridColumn: 'span 6' }}>
                  <div className="card-header">
                    <h2 className="card-title">Spending History</h2>
                  </div>
                  <div className="card-body" style={{ height: '260px' }}>
                    <MonthlySpentChart totalSpent={totalSpent} />
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 6' }}>
                  <div className="card-header">
                    <h2 className="card-title">Category breakdown</h2>
                  </div>
                  <div className="card-body" style={{ height: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <CategoryChart byCategory={categoriesBreakdown} />
                  </div>
                </div>
              </div>

              {/* 5 & 7. Forecasting Section & Goal tracking */}
              <div className="dashboard-grid">
                <div className="card" id="forecast-target" style={{ gridColumn: 'span 8' }}>
                  <div className="card-header">
                    <h2 className="card-title">Spending Forecasting</h2>
                  </div>
                  <div className="card-body" style={{ height: '260px' }}>
                    <ForecastChart forecastData={forecast || null} />
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <h2 className="card-title">Goal Milestones</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '15px', justifyContent: 'center' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '4px' }}>
                        <span>Emergency Fund</span>
                        <span>₹50,000 / ₹1,00,000</span>
                      </div>
                      <div className="budget-progress" style={{ margin: 0 }}>
                        <div className="budget-progress-bar" style={{ width: '50%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '4px' }}>
                        <span>Vacation Fund</span>
                        <span>₹20,000 / ₹50,000</span>
                      </div>
                      <div className="budget-progress" style={{ margin: 0 }}>
                        <div className="budget-progress-bar" style={{ width: '40%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 6. AI recommendations / copilot panel */}
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">AI Financial Copilot Panel</h2>
                </div>
                <div className="card-body" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
                  <div>
                    <div id="copilot-suggestions" style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '14px' }}>
                      {getSuggestedPrompts('analytics').map((prompt, pIdx) => (
                        <button key={pIdx} type="button" className="header-btn" onClick={() => handleSendCopilotMessage(prompt)} style={{ fontSize: '11px', padding: '6px 10px' }}>
                          {prompt}
                        </button>
                      ))}
                    </div>

                    <div id="copilot-history" ref={copilotHistoryRef} style={{ minHeight: '150px', maxHeight: '250px', overflowY: 'auto', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '10px' }}>
                      {copilotMessages.map((msg, i) => (
                        <div key={i} className={`ai-message ${msg.sender === 'user' ? 'user' : 'bot'} ${msg.sender === 'error' ? 'error' : ''}`} style={{ maxWidth: '90%' }}>
                          {msg.text}
                        </div>
                      ))}
                    </div>

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input 
                        type="text"
                        placeholder="Ask copilot about budgets, recommendations, scoring..."
                        value={copilotInput}
                        onChange={(e) => setCopilotInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSendCopilotMessage();
                        }}
                        className="ai-input"
                      />
                      <button className="header-btn primary" onClick={() => handleSendCopilotMessage()}>
                        Send
                      </button>
                    </div>
                  </div>
                  {/* Right: Cohorts Segmentation peer statistics */}
                  <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: '20px' }}>
                    <h3 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px' }}>Segmentation Cohorts</h3>
                    {clustering ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <p style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                          You are grouped into **Cohort {clustering.current_user_cluster + 1}** based on behaviors.
                        </p>
                        {[0, 1, 2].map((cid) => {
                          const peers = clustering.clusters.filter(u => u.cluster === cid);
                          const isCurrent = clustering.current_user_cluster === cid;
                          return (
                            <div key={cid} style={{ padding: '8px 10px', borderRadius: '6px', border: isCurrent ? '1px solid var(--primary)' : '1px solid var(--border)', backgroundColor: isCurrent ? 'var(--primary-glow)' : 'transparent', fontSize: '11px' }}>
                              <strong>Cohort {cid + 1}</strong>
                              <span style={{ float: 'right', opacity: 0.6 }}>{peers.length} users</span>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p style={{ opacity: 0.6, fontSize: '12px' }}>Loading cohorts...</p>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* 3. Investments Tab */}
          {activeSection === 'investments' && (
            <section className="content-section" id="section-investments">
              {/* Portfolio overview stats */}
              <div className="overview-metrics" style={{ borderBottom: 'none', padding: 0 }}>
                <div className="metric-card">
                  <div className="metric-label">Total Investment</div>
                  <div className="metric-value">
                    ₹{portfolioValuation ? Math.round(portfolioValuation.total_investment).toLocaleString('en-IN') : '1,50,000'}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Current Value</div>
                  <div className="metric-value">
                    ₹{portfolioValuation ? Math.round(portfolioValuation.current_value).toLocaleString('en-IN') : '1,62,600'}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Portfolio returns</div>
                  <div className="metric-value" style={{ color: 'var(--success)' }}>
                    +₹{portfolioValuation ? Math.round(portfolioValuation.absolute_gain).toLocaleString('en-IN') : '12,600'}{' '}
                    ({portfolioValuation ? portfolioValuation.gain_percentage : '8.4'}%)
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Risk Category</div>
                  <div className="metric-value">{investRisk} Risk</div>
                </div>
              </div>

              {/* Allocation & Risk selectors */}
              <div className="dashboard-grid">
                <div className="card" style={{ gridColumn: 'span 6' }}>
                  <div className="card-header">
                    <h2 className="card-title">Portfolio Allocation</h2>
                  </div>
                  <div className="card-body" style={{ height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <AllocationChart riskProfile={user?.risk_profile || 'medium'} />
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 6' }}>
                  <div className="card-header">
                    <h2 className="card-title">Investment Profile Settings</h2>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <div className="form-row" style={{ margin: 0 }}>
                      <label>Primary Goal
                        <select value={investGoal} onChange={(e) => setInvestGoal(e.target.value)}>
                          <option value="Growth">Growth</option>
                          <option value="Income">Income</option>
                          <option value="Balanced">Balanced</option>
                          <option value="ESG">ESG / Sustainable</option>
                        </select>
                      </label>
                      <label>Risk Level
                        <select value={investRisk} onChange={(e) => setInvestRisk(e.target.value)}>
                          <option value="Low">Low</option>
                          <option value="Medium">Medium</option>
                          <option value="High">High</option>
                        </select>
                      </label>
                    </div>
                    <div className="form-row" style={{ margin: 0 }}>
                      <label>Time Horizon
                        <select value={investHorizon} onChange={(e) => setInvestHorizon(e.target.value)}>
                          <option value="Short">0-3 years</option>
                          <option value="Medium">3-7 years</option>
                          <option value="Long">7+ years</option>
                        </select>
                      </label>
                      <label>Sector Preference
                        <select value={investSector} onChange={(e) => setInvestSector(e.target.value)}>
                          <option value="Any">Any Sector</option>
                          <option value="Tech">Tech</option>
                          <option value="Healthcare">Healthcare</option>
                          <option value="RealEstate">Real Estate</option>
                          <option value="Energy">Energy / Clean</option>
                        </select>
                      </label>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
                      <button className="header-btn" onClick={() => setRiskQuizModalOpen(true)}>
                        <span className="material-icons">quiz</span> Take Risk Quiz
                      </button>
                      <button className="header-btn primary" onClick={() => (window as any).showSuccess('Investment profile updated!')}>
                        Save Profile
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Fund listings */}
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">AI Recommended Mutual Funds</h2>
                  <div className="card-actions">
                    <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)} style={{ padding: '4px 8px', fontSize: '12px' }}>
                      <option value="all">All Risks</option>
                      <option value="low">Low Risk</option>
                      <option value="medium">Medium Risk</option>
                      <option value="high">High Risk</option>
                    </select>
                    <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} style={{ padding: '4px 8px', fontSize: '12px' }}>
                      <option value="default">Default Sort</option>
                      <option value="risk">Sort by Risk</option>
                      <option value="investment">Min. Investment</option>
                      <option value="name">Sort by Name</option>
                    </select>
                  </div>
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                  <div className="opportunities-grid" style={{ padding: '20px' }}>
                    {getFilteredRecs().map((rec, idx) => {
                      const isEquity = rec.scheme.toLowerCase().includes('equity') || rec.scheme.toLowerCase().includes('stock');
                      const isBond = rec.scheme.toLowerCase().includes('bond') || rec.scheme.toLowerCase().includes('debt') || rec.scheme.toLowerCase().includes('fixed');
                      const type = isEquity ? 'high' : isBond ? 'low' : 'medium';
                      const icon = isEquity ? 'trending_up' : isBond ? 'account_balance' : 'savings';

                      return (
                        <div key={idx} className="opportunity-card">
                          <div className="opportunity-header">
                            <div className="opportunity-icon">
                              <span className="material-icons">{icon}</span>
                            </div>
                            <span className={`chip ${type}`}>{type.toUpperCase()}</span>
                          </div>
                          <div className="opportunity-content">
                            <h4 style={{ fontSize: '13px', fontWeight: 600 }}>{rec.scheme}</h4>
                            <p style={{ fontSize: '12px', opacity: 0.8, marginTop: '4px', minHeight: '40px' }}>{rec.reason}</p>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginTop: '10px', opacity: 0.8 }}>
                              <span>Min Investment:</span>
                              <strong>₹{isEquity ? '1,000' : isBond ? '5,000' : '500'}</strong>
                            </div>
                          </div>
                          <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                            <button className="header-btn primary" style={{ flex: 1, padding: '6px', fontSize: '11px', justifyContent: 'center' }} onClick={() => (window as any).showToast('Details coming soon', 'info')}>Details</button>
                            <button className="header-btn" style={{ flex: 1, padding: '6px', fontSize: '11px', justifyContent: 'center' }} onClick={() => (window as any).showSuccess('Added to Watchlist!')}>Watchlist</button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Stock Price visualizer */}
              <div className="dashboard-grid" style={{ marginTop: '25px' }}>
                <div className="card" style={{ gridColumn: 'span 8' }}>
                  <div className="card-header">
                    <h2 className="card-title">Stock Price Trends</h2>
                    <div className="card-actions" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <select 
                        value={selectedStock} 
                        onChange={(e) => setSelectedStock(e.target.value)}
                        style={{ padding: '6px 12px', fontSize: '13px', borderRadius: '8px', border: '1px solid var(--border)' }}
                      >
                        {stockSymbols.length > 0 ? (
                          stockSymbols.map((sym: string) => (
                            <option key={sym} value={sym}>{sym}</option>
                          ))
                        ) : (
                          <option value="AAPL">AAPL</option>
                        )}
                      </select>
                    </div>
                  </div>
                  <div className="card-body" style={{ height: '280px' }}>
                    {stockLoading ? (
                      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', opacity: 0.6 }}>
                        Loading chart data...
                      </div>
                    ) : (
                      <StockPriceChart stockData={stockHistory} symbol={selectedStock} />
                    )}
                  </div>
                </div>

                <div className="card" style={{ gridColumn: 'span 4' }}>
                  <div className="card-header">
                    <h2 className="card-title">Available Tickers</h2>
                  </div>
                  <div className="card-body" style={{ overflowY: 'auto', maxHeight: '280px', padding: '15px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                      {stockSymbols.map((sym: string) => (
                        <button
                          key={sym}
                          onClick={() => setSelectedStock(sym)}
                          className={`header-btn ${selectedStock === sym ? 'primary' : ''}`}
                          style={{ padding: '8px', fontSize: '12px', justifyContent: 'center', borderRadius: '6px' }}
                        >
                          {sym}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* 4. Transactions Tab (Fully Featured Table, Search, Category Chips, Pagination, Bulk Actions) */}
          {activeSection === 'transactions' && (
            <section className="content-section" id="section-transactions">
              <div className="card">
                <div className="card-header" style={{ borderBottom: 'none' }}>
                  <h2 className="card-title">All Transactions</h2>
                  <div className="card-actions" style={{ gap: '10px' }}>
                    {txSelectedIds.length > 0 && (
                      <button className="header-btn" onClick={handleBulkDelete} style={{ color: 'var(--danger)', borderColor: 'var(--danger-border)', backgroundColor: 'var(--danger-bg)' }}>
                        Delete Selected ({txSelectedIds.length})
                      </button>
                    )}
                    <button className="header-btn primary" onClick={() => setExpenseModalOpen(true)}>
                      <span className="material-icons">add</span> Add Expense
                    </button>
                  </div>
                </div>

                {/* Filters Row */}
                <div style={{ display: 'flex', gap: '15px', padding: '0 20px 15px', borderBottom: '1px solid var(--border)', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: '200px' }}>
                    <span className="material-icons" style={{ color: 'var(--text-muted)', fontSize: '20px' }}>search</span>
                    <input 
                      type="text" 
                      placeholder="Search transactions..." 
                      value={txSearchQuery} 
                      onChange={(e) => { setTxSearchQuery(e.target.value); setTxCurrentPage(1); }}
                      className="ai-input" 
                      style={{ margin: 0, padding: '6px 12px' }}
                    />
                  </div>
                  <select 
                    value={txCategoryFilter} 
                    onChange={(e) => { setTxCategoryFilter(e.target.value); setTxCurrentPage(1); }} 
                    style={{ padding: '6px 12px', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '13px' }}
                  >
                    <option value="all">All Categories</option>
                    <option value="Housing">Housing</option>
                    <option value="Food">Food</option>
                    <option value="Transportation">Transportation</option>
                    <option value="Utilities">Utilities</option>
                    <option value="Entertainment">Entertainment</option>
                    <option value="Savings">Savings</option>
                  </select>
                </div>

                {/* Transactions Table */}
                <div className="card-body" style={{ padding: 0 }}>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                      <thead>
                        <tr style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border)' }}>
                          <th style={{ padding: '12px 20px', width: '40px' }}>
                            <input 
                              type="checkbox"
                              checked={paginatedTxs.length > 0 && paginatedTxs.every(t => txSelectedIds.includes(t.id || ''))}
                              onChange={() => toggleSelectAllTxs(paginatedTxs)}
                              style={{ cursor: 'pointer' }}
                            />
                          </th>
                          <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>Date</th>
                          <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>Description</th>
                          <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>Category</th>
                          <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'right' }}>Amount</th>
                          <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'center', width: '80px' }}>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedTxs.length > 0 ? (
                          paginatedTxs.map((exp, i) => {
                            const isSelected = txSelectedIds.includes(exp.id || '');
                            const catLower = exp.category.toLowerCase();
                            const getBadgeClass = (cat: string) => {
                              if (['food', 'dining'].includes(cat)) return 'category-badge food';
                              if (['bills', 'utilities', 'housing', 'rent', 'transportation'].includes(cat)) return 'category-badge bills';
                              if (['shopping', 'clothing', 'supplies'].includes(cat)) return 'category-badge shopping';
                              if (['entertainment', 'leisure', 'recreation'].includes(cat)) return 'category-badge entertainment';
                              return 'category-badge default';
                            };
                            return (
                              <tr key={exp.id || i} style={{ borderBottom: '1px solid var(--border)', backgroundColor: isSelected ? 'rgba(79, 70, 229, 0.04)' : 'transparent' }}>
                                <td style={{ padding: '12px 20px' }}>
                                  <input 
                                    type="checkbox" 
                                    checked={isSelected}
                                    onChange={() => toggleSelectTx(exp.id || '')}
                                    style={{ cursor: 'pointer' }}
                                    aria-label={`Select transaction ${exp.description || exp.category}`}
                                  />
                                </td>
                                <td style={{ padding: '12px 20px', color: 'var(--text-secondary)' }}>{exp.date}</td>
                                <td style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-primary)' }}>{exp.description || exp.category}</td>
                                <td style={{ padding: '12px 20px' }}>
                                  <span className={getBadgeClass(catLower)}>{exp.category}</span>
                                </td>
                                <td style={{ padding: '12px 20px', textAlign: 'right', fontWeight: 700, fontFamily: 'var(--font-heading)' }}>
                                  ₹{Number(exp.amount).toFixed(2)}
                                </td>
                                <td style={{ padding: '12px 20px', textAlign: 'center' }}>
                                  <button
                                    onClick={() => {
                                      if (window.confirm('Are you sure you want to delete this transaction?')) {
                                        deleteExpenseMutation.mutate(exp.id, {
                                          onSuccess: () => (window as any).showSuccess('Transaction deleted successfully')
                                        });
                                      }
                                    }}
                                    style={{ border: 'none', background: 'none', color: 'var(--danger)', cursor: 'pointer', padding: '4px' }}
                                    title="Delete Transaction"
                                    aria-label={`Delete transaction ${exp.description || exp.category}`}
                                  >
                                    <span className="material-icons" style={{ fontSize: '18px' }}>delete</span>
                                  </button>
                                </td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr>
                            <td colSpan={6}>
                              <div className="empty-state-panel">
                                <span className="material-icons empty-state-icon">receipt_long</span>
                                <div className="empty-state-text">No transactions match your search filter</div>
                                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Try resetting the search terms or category selections to view your history.</p>
                              </div>
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination row */}
                  {filteredTxs.length > txItemsPerPage && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '15px 20px', borderTop: '1px solid var(--border)', fontSize: '12px' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>
                        Showing {txStartIndex + 1}-{Math.min(txStartIndex + txItemsPerPage, filteredTxs.length)} of {filteredTxs.length} entries
                      </span>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button 
                          className="header-btn" 
                          disabled={txCurrentPage === 1}
                          onClick={() => setTxCurrentPage(p => p - 1)}
                          style={{ padding: '4px 8px', fontSize: '11px' }}
                        >
                          Previous
                        </button>
                        <button 
                          className="header-btn" 
                          disabled={txCurrentPage === totalTxPages}
                          onClick={() => setTxCurrentPage(p => p + 1)}
                          style={{ padding: '4px 8px', fontSize: '11px' }}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}
          
          {/* 5. Budget Tab */}
          {activeSection === 'budget' && (
            <section className="content-section" id="section-budget">
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">Budget Categories Utilization</h2>
                  <div className="card-actions">
                    <button className="header-btn primary" onClick={() => setBudgetModalOpen(true)}>
                      <span className="material-icons">edit</span> Configure Budgets
                    </button>
                  </div>
                </div>
                <div className="card-body">
                  <div className="budget-summary" id="budget-summary">
                    <div className="budget-total" style={{ marginBottom: '20px', fontSize: '14px', borderBottom: '1px solid var(--border)', paddingBottom: '10px' }}>
                      <strong>Global Monthly Limit:</strong>{' '}
                      <span id="budget-total-amount" style={{ color: 'var(--primary)', fontWeight: 700, marginLeft: '5px' }}>
                        {totalBudgetLimit > 0 ? `₹${totalBudgetLimit.toLocaleString('en-IN')}` : 'Not set'}
                      </span>
                    </div>

                    {budgetSplit ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                        {(['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Savings'] as const).map((cat) => {
                          const limit = budgetSplit[cat] || 0;
                          const spent = categoriesBreakdown[cat.toLowerCase()] || categoriesBreakdown[cat] || 0;
                          const remaining = limit - spent;
                          const pct = limit > 0 ? (spent / limit) * 100 : 0;
                          
                          let barColor = 'var(--primary)';
                          let statusText = remaining >= 0 ? `₹${remaining.toLocaleString('en-IN')} remaining` : `₹${Math.abs(remaining).toLocaleString('en-IN')} OVER BUDGET`;
                          let statusColor = remaining >= 0 ? 'var(--text-secondary)' : 'var(--danger)';

                          if (pct >= 100) {
                            barColor = 'var(--danger)';
                          } else if (pct >= 80) {
                            barColor = 'var(--warning, #f59e0b)';
                          } else if (pct < 50) {
                            barColor = 'var(--success, #10b981)';
                          }

                          return (
                            <div key={cat} style={{ backgroundColor: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: '8px', border: pct >= 100 ? '1px solid var(--danger-border, #fecaca)' : '1px solid var(--border)' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <strong style={{ fontSize: '13px' }}>{cat}</strong>
                                  {pct >= 100 && <span style={{ fontSize: '10px', backgroundColor: 'var(--danger)', color: 'white', padding: '2px 6px', borderRadius: '10px', fontWeight: 600 }}>OVER LIMIT</span>}
                                  {pct >= 80 && pct < 100 && <span style={{ fontSize: '10px', backgroundColor: 'var(--warning, #f59e0b)', color: 'white', padding: '2px 6px', borderRadius: '10px', fontWeight: 600 }}>NEAR LIMIT</span>}
                                </div>
                                <div style={{ fontSize: '12px', fontWeight: 700, color: statusColor }}>
                                  {statusText}
                                </div>
                              </div>
                              <div className="budget-progress" style={{ margin: '0 0 6px 0', height: '8px', backgroundColor: 'var(--bg-tertiary, #e2e8f0)' }}>
                                <div style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: barColor, height: '100%', borderRadius: '4px', transition: 'width 0.3s ease' }}></div>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                                <span>Spent: ₹{spent.toLocaleString('en-IN')}</span>
                                <span>Budget: ₹{limit.toLocaleString('en-IN')}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div style={{ padding: '30px 20px', textAlign: 'center', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', margin: '10px 0' }}>
                        <span className="material-icons" style={{ fontSize: '32px', color: 'var(--text-muted)', marginBottom: '10px', display: 'block' }}>account_balance_wallet</span>
                        <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>No budgets configured</h4>
                        <p style={{ opacity: 0.7, fontSize: '12px', lineHeight: 1.5, marginBottom: '15px' }}>
                          Configure your first budget envelope to track spending.
                        </p>
                        <button className="header-btn primary" onClick={() => setBudgetModalOpen(true)} style={{ margin: '0 auto', fontSize: '12px', padding: '6px 12px' }}>
                          Configure Budgets
                        </button>
                      </div>
                    )}

                    <div className="rewards-summary" style={{ marginTop: '30px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                      <h3 className="rewards-title" style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px' }}>Achieved Badges</h3>
                      <div className="rewards-list" id="rewards-list" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {budgetSplit && totalBudgetLimit > 0 ? (
                          (() => {
                            const badges = [];
                            const ratio = totalSpent / totalBudgetLimit;
                            const pctTotal = Math.round(ratio * 100);

                            if (pctTotal <= 50) {
                              badges.push(<span key="bronze" className="chip low">Budget Bronze: &lt; 50% spent</span>);
                            }
                            if (pctTotal <= 75) {
                              badges.push(<span key="saver" className="chip low">Saver Streak: Under target</span>);
                            }
                            if (pctTotal >= 90 && pctTotal < 100) {
                              badges.push(<span key="caution" className="chip medium">Caution: 90% of budget</span>);
                            }
                            if (pctTotal >= 100) {
                              badges.push(<span key="over" className="chip high">Over Limit: Review expenses</span>);
                            }

                            // Category progress checks
                            Object.keys(budgetSplit).forEach((c) => {
                              const lim = (budgetSplit as any)[c] || 0;
                              const spt = categoriesBreakdown[c.toLowerCase()] || categoriesBreakdown[c] || 0;
                              if (lim === 0) return;
                              const p = Math.round((spt / lim) * 100);
                              if (p <= 60) {
                                badges.push(<span key={c} className="chip low">{c}: Great control ({p}%)</span>);
                              } else if (p >= 100) {
                                badges.push(<span key={c} className="chip high">{c}: Over limit ({p}%)</span>);
                              }
                            });

                            if (badges.length === 0) {
                              return <p style={{ opacity: 0.6 }}>Track your expenses to earn reward badges.</p>;
                            }

                            return badges.slice(0, 6);
                          })()
                        ) : (
                          <span className="chip medium">Track spending to unlock rewards!</span>
                        )}
                      </div>
                    </div>

                    {/* AI Budget Nudge & Tips */}
                    <div style={{ marginTop: '30px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                      <h3 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span className="material-icons" style={{ color: 'var(--primary)', fontSize: '18px' }}>lightbulb</span>
                        AI Budget Advisory
                      </h3>
                      <div style={{ padding: '12px 16px', borderRadius: '8px', backgroundColor: '#f3f0ff', border: '1px solid #e5dbff', fontSize: '13px', lineHeight: 1.5, color: '#3b306b' }}>
                        {tipsLoading ? (
                          <span style={{ opacity: 0.6 }}>Loading advisory...</span>
                        ) : budgetTipData?.tip ? (
                          <span>{budgetTipData.tip}</span>
                        ) : (
                          <span style={{ opacity: 0.6 }}>No active budget tips at this time.</span>
                        )}
                      </div>
                    </div>

                  </div>
                </div>
              </div>
            </section>
          )}
          
          {/* 6. Settings Tab (Modernized Form Fields) */}
          {activeSection === 'settings' && (
            <section className="content-section" id="section-settings">
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">Profile & Preferences</h2>
                </div>
                <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                  {/* Account detail columns */}
                  <h3 style={{ fontSize: '13px', fontWeight: 600, borderBottom: '1px solid var(--border)', paddingBottom: '6px' }}>User Details</h3>
                  <div className="form-row" style={{ margin: 0 }}>
                    <label>Full Name
                      <input type="text" value={profileName} onChange={(e) => setProfileName(e.target.value)} placeholder="Full Name" />
                    </label>
                    <label>Email Address
                      <input type="email" value={profileEmail} onChange={(e) => setProfileEmail(e.target.value)} placeholder="Email Address" />
                    </label>
                  </div>
                  <div className="form-row" style={{ margin: 0 }}>
                    <label>Monthly Income (₹)
                      <input type="number" value={profileIncome} onChange={(e) => setProfileIncome(Number(e.target.value))} placeholder="Income" />
                    </label>
                    <label>Investment Goal
                      <select value={profileGoal} onChange={(e) => setProfileGoal(e.target.value)}>
                        <option value="Growth">Growth Focus</option>
                        <option value="Income">Income Focus</option>
                        <option value="Balanced">Balanced Mix</option>
                      </select>
                    </label>
                  </div>

                  <h3 style={{ fontSize: '13px', fontWeight: 600, borderBottom: '1px solid var(--border)', paddingBottom: '6px', marginTop: '10px' }}>App Settings</h3>
                  <div className="form-row" style={{ margin: 0 }}>
                    <label>App Theme
                      <select value={appTheme} onChange={(e) => setAppTheme(e.target.value)}>
                        <option value="light">Light (Default)</option>
                        <option value="dark">Dark Theme</option>
                        <option value="ocean">Ocean Blue</option>
                        <option value="sunset">Sunset Orange</option>
                      </select>
                    </label>
                    <label>Risk Tolerance
                      <select value={profileRisk} onChange={(e) => setProfileRisk(e.target.value)}>
                        <option value="low">Conservative</option>
                        <option value="medium">Moderate</option>
                        <option value="high">Aggressive</option>
                      </select>
                    </label>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '10px', marginTop: '20px', justifyContent: 'flex-end' }}>
                    <button className="header-btn" onClick={() => handleRefreshData()}>Reset</button>
                    <button className="header-btn primary" onClick={handleSaveProfile}>Save Changes</button>
                  </div>
                </div>
              </div>
            </section>
          )}

        </main>
      </div>

      <DemoTour setActiveSection={setActiveSection} />

      {/* Floating AI Panel Toggler */}
      <button 
        className="floating-ai-btn" 
        id="floating-ai-toggle" 
        onClick={() => setFloatingAiOpen(!floatingAiOpen)}
        aria-label="Open Nexus AI Assistant"
      >
        <span className="material-icons">smart_toy</span>
      </button>

      {/* Floating AI Panel Drawer */}
      <div className={`floating-ai-drawer ${floatingAiOpen ? 'open' : ''}`} id="floating-ai-panel">
        <div className="ai-header">
          <div className="ai-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="material-icons" style={{ color: 'var(--primary)', fontSize: '18px' }}>auto_awesome</span> 
            Nexus AI
          </div>
          <button className="card-action-btn" id="floating-ai-close" onClick={() => setFloatingAiOpen(false)} style={{ marginLeft: 'auto' }}>
            <span className="material-icons">close</span>
          </button>
        </div>
        
        <div className="ai-chat" id="floating-ai-body" ref={floatingHistoryRef}>
          {floatingAiMessages.map((msg, idx) => {
            if (msg.isError) {
              return (
                <div key={idx} className="ai-message bot" style={{ maxWidth: '90%', border: '1px solid var(--danger-border)', backgroundColor: 'var(--danger-bg)', color: 'var(--danger)', padding: '10px 12px', borderRadius: '8px', margin: '4px 0' }}>
                  <div>{msg.text}</div>
                  {msg.originalText && (
                    <button 
                      className="retry-btn" 
                      onClick={() => handleSendFloatingAiMessage(msg.originalText)}
                    >
                      <span className="material-icons" style={{ fontSize: '12px' }}>replay</span>
                      Retry
                    </button>
                  )}
                </div>
              );
            }
            return (
              <div key={idx} className={`ai-message ${msg.sender === 'user' ? 'user' : 'bot'}`} style={{ maxWidth: '95%' }}>
                {msg.sender === 'user' ? <p style={{ margin: 0 }}>{msg.text}</p> : formatMessageText(msg.text)}
              </div>
            );
          })}
          {floatingAiLoading && (
            <div className="ai-message bot" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div className="typing-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
        </div>

        {/* Suggested prompts list */}
        <div className="suggested-prompts">
          <span style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '2px' }}>Context Suggested Questions</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {getSuggestedPrompts(activeSection).map((prompt, pIdx) => (
              <button 
                key={pIdx} 
                className="prompt-pill" 
                onClick={() => handleSendFloatingAiMessage(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        <div className="ai-input-container">
          <input 
            type="text" 
            placeholder={`Ask about ${activeSection === 'dashboard' ? 'finances' : activeSection}...`} 
            value={floatingAiInput}
            onChange={(e) => setFloatingAiInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSendFloatingAiMessage();
            }}
            className="ai-input"
            disabled={floatingAiLoading}
          />
          <button 
            className="ai-send-btn" 
            onClick={() => handleSendFloatingAiMessage()}
            disabled={floatingAiLoading}
          >
            <span className="material-icons">send</span>
          </button>
        </div>
      </div>

      {/* Controlled Modals */}
      <ExpenseModal 
        isOpen={expenseModalOpen} 
        onClose={() => setExpenseModalOpen(false)}
        onSuccess={handleRefreshData}
      />
      <BudgetModal 
        isOpen={budgetModalOpen}
        onClose={() => setBudgetModalOpen(false)}
        onSuccess={handleRefreshData}
      />
      <RiskQuizModal 
        isOpen={riskQuizModalOpen}
        onClose={() => setRiskQuizModalOpen(false)}
        onComplete={handleQuizComplete}
      />
    </div>
  );
};
