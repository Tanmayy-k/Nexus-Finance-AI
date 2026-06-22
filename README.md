# Nexus Finance – AI-Powered Personal Finance & Investment Advisor

Nexus Finance is a full-stack fintech application that helps users manage expenses, track budgets, improve financial health, and receive AI-powered financial guidance through Machine Learning and Generative AI.

The platform combines financial analytics, forecasting, anomaly detection, explainable AI, and a Gemini-powered financial copilot to deliver personalized financial recommendations.

---

## 🚀 Key Features

### Authentication & Security

* JWT-based authentication
* Secure password hashing
* Protected API routes
* Environment-based configuration management

### Financial Management

* Expense tracking and categorization
* Income and budget management
* Personalized budgeting recommendations
* Spending summaries and analytics
* Real-time financial dashboard

### Investment & Recommendations

* Personalized stock and mutual fund recommendations
* User segmentation using behavioral clustering
* Risk-aware investment suggestions

---

## 🧠 AI & Machine Learning Features

### Financial Health Scoring

* Random Forest-based Financial Health Score
* Evaluates income, expenses, investments, and savings behavior
* Provides a personalized financial score

### Explainable AI (SHAP)

* SHAP-based model explainability
* Identifies positive and negative factors affecting the user's score
* Transparent ML decision-making

### User Segmentation

* K-Means Clustering
* Groups users based on financial behavior, goals, and risk profile
* Powers recommendation personalization

### Spending Forecasting

* Predicts future monthly spending trends
* Identifies increasing or decreasing expense patterns
* Provides proactive financial planning insights

### Anomaly Detection

* Isolation Forest-based anomaly detection
* Detects unusual spending activity
* Highlights potentially risky financial behavior

### AI Insights Engine

* Generates personalized financial insights
* Combines forecasting, anomalies, spending trends, and scoring data
* Provides actionable recommendations

### Gemini Financial Copilot

* Gemini 2.5 Flash integration
* Context-aware financial assistant
* Uses:

  * Financial Health Score
  * SHAP factors
  * Forecast summaries
  * Spending behavior
  * Anomaly insights
* Supports intelligent financial coaching and planning

---

## 🛠️ Technology Stack

### Backend

* Python
* Flask
* SQLAlchemy
* Alembic

### Database

* SQLite (Current)
* PostgreSQL / Supabase Ready

### Machine Learning & AI

* Scikit-learn
* Pandas
* SHAP
* Random Forest
* K-Means Clustering
* Isolation Forest
* Google Gemini 2.5 Flash

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

---

## 📊 System Architecture

User
↓
Frontend Dashboard
↓
Flask REST APIs
↓
Business Services
↓
Machine Learning Layer
(Random Forest, SHAP, K-Means, Isolation Forest)
↓
Gemini Financial Copilot
↓
SQLite / PostgreSQL

---

## ⚙️ Local Setup

### Backend

```bash
cd backend

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

Backend runs on:

```text
http://127.0.0.1:5000
```

### Frontend

```bash
cd frontend

python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500/login.html
```

---

## 🔐 Environment Variables

Create a `.env` file inside the backend directory:

```env
JWT_SECRET_KEY=your_jwt_secret
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///finance.db

# Optional
GEMINI_API_KEY=your_gemini_api_key
```

---

## 📌 Current Status

### Completed

* Authentication System
* Expense Tracking
* Budget Management
* Financial Health Scoring
* SHAP Explainability
* User Segmentation
* Spending Forecasting
* Anomaly Detection
* AI Insights Engine
* Gemini Financial Copilot
* PostgreSQL Migration Support
* React Frontend Migration
* Supabase PostgreSQL Migration

### Planned
* Cloud Deployment
* Enhanced Analytics Dashboard

---

## 👨‍💻 Contributors

Developed as an AI-powered fintech project focused on Machine Learning, Explainable AI, Financial Analytics, and Generative AI integration.
