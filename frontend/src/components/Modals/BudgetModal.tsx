import React, { useState, useEffect } from 'react';
import { useBudgetStore } from '../../store/useBudgetStore';
import { useAuthStore } from '../../store/useAuthStore';

interface BudgetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

const CATEGORIES = ['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Savings'];

export const BudgetModal: React.FC<BudgetModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const { createOrUpdateBudget, budgetSplit } = useBudgetStore();
  const { setProfile, user } = useAuthStore();

  const [income, setIncome] = useState('');
  const [goal, setGoal] = useState('Financial Stability');
  const [risk, setRisk] = useState('medium');
  
  // Custom Category Limits
  const [customSplit, setCustomSplit] = useState<Record<string, string>>({
    Housing: '',
    Food: '',
    Transportation: '',
    Utilities: '',
    Entertainment: '',
    Savings: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setIncome(user?.income?.toString() || '');
      setGoal(user?.goal || 'Financial Stability');
      setRisk(user?.risk_profile || 'medium');
      
      if (budgetSplit) {
        const prefilled: Record<string, string> = {};
        CATEGORIES.forEach(cat => {
          prefilled[cat] = budgetSplit[cat]?.toString() || '';
        });
        setCustomSplit(prefilled);
      }
    }
  }, [isOpen, user, budgetSplit]);

  if (!isOpen) return null;

  const totalAllocated = Object.values(customSplit).reduce((acc, val) => acc + (parseFloat(val) || 0), 0);
  const incomeVal = parseFloat(income) || 0;
  const remaining = incomeVal - totalAllocated;

  const handleCategoryChange = (category: string, value: string) => {
    setCustomSplit(prev => ({ ...prev, [category]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    if (totalAllocated > incomeVal) {
      (window as any).showToast('You have allocated more than your monthly income.', 'error');
      return;
    }

    setIsSubmitting(true);

    try {
      const incVal = parseFloat(income);
      
      const formattedSplit: Record<string, number> = {};
      Object.keys(customSplit).forEach(key => {
        formattedSplit[key] = parseFloat(customSplit[key]) || 0;
      });

      await createOrUpdateBudget(incVal, goal, risk, formattedSplit);
      setProfile({ income: incVal, goal, risk_profile: risk });
      (window as any).showToast('Budget created successfully!', 'success');
      
      onSuccess();
      onClose();
    } catch (error: any) {
      (window as any).showToast('Failed to save budget settings', 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal" onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div className="modal-content" style={{ maxWidth: '600px' }}>
        <div className="modal-header">
          <h3>Custom Zero-Based Budget</h3>
          <span className="close" onClick={onClose}>&times;</span>
        </div>
        <div className="modal-body">
          <form id="budget-form" onSubmit={handleSubmit}>
            <div className="form-row">
              <label>Monthly Income
                <input 
                  type="number" 
                  id="budget-income" 
                  placeholder="50000" 
                  value={income}
                  onChange={(e) => setIncome(e.target.value)}
                  required 
                />
              </label>
              <label>Financial Goal
                <select 
                  id="budget-goal" 
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  required
                >
                  <option value="Financial Stability">Financial Stability</option>
                  <option value="Emergency Fund">Emergency Fund</option>
                  <option value="Investment Growth">Investment Growth</option>
                  <option value="Debt Payoff">Debt Payoff</option>
                  <option value="Retirement Planning">Retirement Planning</option>
                </select>
              </label>
            </div>
            
            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h4 style={{ margin: 0, fontSize: '14px' }}>Category Allocation</h4>
                <div style={{ fontSize: '13px', fontWeight: 600, color: remaining < 0 ? 'var(--danger)' : remaining === 0 ? 'var(--success)' : 'var(--text-secondary)' }}>
                  Remaining: ₹{remaining.toLocaleString()}
                </div>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                {CATEGORIES.map(category => (
                  <div key={category}>
                    <label style={{ fontSize: '12px', marginBottom: '4px', display: 'block' }}>{category}</label>
                    <input 
                      type="number" 
                      placeholder="0"
                      value={customSplit[category] || ''}
                      onChange={(e) => handleCategoryChange(category, e.target.value)}
                      style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid var(--border)', backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="form-actions" style={{ display: 'flex', gap: '10px', marginTop: '20px', justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={isSubmitting || totalAllocated > incomeVal}>
                {isSubmitting ? 'Saving...' : 'Save Budget'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
