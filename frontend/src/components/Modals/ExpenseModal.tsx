import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';

interface ExpenseModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const ExpenseModal: React.FC<ExpenseModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [date, setDate] = useState('');
  const [amount, setAmount] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('Housing');
  const [paymentMethod, setPaymentMethod] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (isOpen) {
      // Pre-fill with today's date in YYYY-MM-DD
      const today = new Date().toISOString().split('T')[0];
      setDate(today);
      setAmount('');
      setDescription('');
      setCategory('Housing');
      setPaymentMethod('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    setIsSubmitting(true);

    try {
      const response = await api.post('/expense', {
        category: category,
        amount: parseFloat(amount),
        date: date,
        description: description,
      });

      const data = response.data;
      if (data.nudge_message) {
        (window as any).showToast(data.nudge_message, 'warning');
      } else {
        (window as any).showToast('Expense added successfully!', 'success');
      }

      onSuccess();
      onClose();
    } catch (error: any) {
      const errMsg = error.response?.data?.error || error.message || 'Failed to add expense';
      (window as any).showToast(errMsg, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal" style={{ display: 'flex' }} onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div className="modal-content">
        <div className="modal-header">
          <h3>Add Expense</h3>
          <span className="close" onClick={onClose}>&times;</span>
        </div>
        <div className="modal-body">
          <form id="expense-form" onSubmit={handleSubmit}>
            <div className="form-row">
              <label>Date
                <input 
                  type="date" 
                  id="expense-date" 
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  required 
                />
              </label>
              <label>Amount
                <input 
                  type="number" 
                  id="expense-amount" 
                  step="0.01" 
                  min="0" 
                  placeholder="0.00" 
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  required 
                />
              </label>
            </div>
            <div className="form-row">
              <label>Description
                <input 
                  type="text" 
                  id="expense-desc" 
                  placeholder="e.g., Groceries" 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  required 
                />
              </label>
              <label>Category
                <select 
                  id="expense-category" 
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  required
                >
                  <option value="Housing">Housing</option>
                  <option value="Food">Food</option>
                  <option value="Transportation">Transportation</option>
                  <option value="Utilities">Utilities</option>
                  <option value="Entertainment">Entertainment</option>
                  <option value="Savings">Savings</option>
                  <option value="Healthcare">Healthcare</option>
                  <option value="Education">Education</option>
                  <option value="Shopping">Shopping</option>
                  <option value="Other">Other</option>
                </select>
              </label>
            </div>
            <div className="form-row">
              <label>Payment Method
                <input 
                  type="text" 
                  id="expense-method" 
                  placeholder="e.g., Visa **** 1234" 
                  value={paymentMethod}
                  onChange={(e) => setPaymentMethod(e.target.value)}
                />
              </label>
            </div>
            <div className="modal-actions">
              <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                {isSubmitting ? 'Adding...' : 'Save Expense'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
