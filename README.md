<div align="center">

# ✦ NEXUS FINANCE AI

### *AI-Powered Personal Finance & Investment Advisor*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://supabase.com)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-AI_Copilot-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![AWS](https://img.shields.io/badge/AWS-Production_Ready-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-8b5cf6?style=flat-square)](CONTRIBUTING.md)
[![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=flat-square)](https://black.readthedocs.io)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vite.dev)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff6b6b?style=flat-square)](https://shap.readthedocs.io)

<br/>

> **A production-grade fintech platform** that fuses Machine Learning, Explainable AI, and Generative AI to deliver personalized financial health scoring, real-time anomaly detection, spending forecasts, and a context-aware Gemini-powered financial copilot — all behind a modern React TypeScript frontend.

<br/>

[**API Reference**](#-api-endpoints) · [**Setup Guide**](#️-local-setup) · [**Architecture**](#-architecture-overview)

</div>

---

## 📋 Table of Contents

- [Why Nexus Finance?](#-why-nexus-finance)
- [Architecture Overview](#-architecture-overview)
- [AI/ML Pipeline](#-aiml-pipeline)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Authentication Flow](#-authentication-flow)
- [Financial Score Pipeline](#-financial-health-score-pipeline)
- [SHAP Explainability](#-shap-explainability)
- [Forecasting Pipeline](#-spending-forecasting-pipeline)
- [Anomaly Detection](#-anomaly-detection)
- [Gemini Copilot](#-gemini-25-flash-copilot)
- [Local Setup](#️-local-setup)
- [Environment Variables](#-environment-variables)
- [PostgreSQL / Supabase Setup](#-postgresql--supabase-setup)
- [AWS Deployment](#️-aws-deployment)
- [Future Roadmap](#-future-roadmap)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

## 🎯 Why Nexus Finance?

Most personal finance apps are either too simple to be actionable or too complex to be usable. Nexus Finance bridges this gap by combining **production-grade ML models** with a **conversational AI copilot** — giving users the financial intelligence of a personal advisor, at zero cost.

### The Problem → Solution

| Problem | Nexus Finance Solution |
|---------|----------------------|
| 📊 Users don't know *why* their finances are unhealthy | **SHAP Explainability** breaks down every score factor in plain language |
| 🤔 Generic budgeting advice ignores personal context | **Gemini 2.5 Flash Copilot** references the user's real income, spending & goals |
| 🚨 Fraudulent or unusual transactions go unnoticed | **Isolation Forest Anomaly Detection** flags unusual expenses in real-time |
| 📈 No visibility into future spending trends | **Time-series Forecasting** predicts next month's spending from historical data |
| 👥 Everyone gets the same recommendations | **K-Means User Segmentation** groups users by financial behavior for personalized advice |
| 🔢 Financial health is an opaque black box | **Random Forest Score (0–100)** with full feature importance transparency |

---

## 🏗 Architecture Overview

```mermaid
graph TB
    subgraph Client["🖥 React Frontend (TypeScript + Vite)"]
        UI[Dashboard / Auth / Analytics]
        STORE[Zustand State Store]
        AXIOS[Axios + JWT Interceptor]
    end

    subgraph Gateway["🔒 Flask API Gateway"]
        AUTH[JWT Auth Middleware]
        ROUTES[Blueprint Routes]
        CORS[CORS Policy]
    end

    subgraph ML["🤖 ML Services Layer"]
        RF[Random Forest Health Score]
        SHAP[SHAP Explainability]
        KMEANS[K-Means Segmentation]
        ISO[Isolation Forest Anomaly Detection]
        FORECAST[Time-Series Forecasting]
    end

    subgraph AI["✨ Generative AI Layer"]
        CTX[Context Builder]
        GEMINI[Gemini 2.5 Flash]
        FALLBACK[Rule-Based Fallback Advisor]
    end

    subgraph Data["🗄 Data Layer"]
        PG[(PostgreSQL Supabase)]
        SQLITE[(SQLite Dev/Local)]
        ALEMBIC[Alembic Migrations]
    end

    UI --> AXIOS --> ROUTES
    ROUTES --> AUTH --> ML
    ROUTES --> AUTH --> AI
    ML --> Data
    AI --> CTX --> GEMINI
    GEMINI -->|"API failure"| FALLBACK
    Data --> PG
    Data --> SQLITE
    ALEMBIC --> PG
```

---

## 🤖 AI/ML Pipeline

```mermaid
flowchart LR
    EXP["💳 User Expenses"] --> FE["Feature Engineering\nIncome / Savings / Spending"]
    FE --> RF["🌲 Random Forest\nHealth Score 0-100"]
    RF --> SHAP["🔍 SHAP Values\nFeature Attribution"]
    SHAP --> SPLIT["Positive / Negative\nScore Drivers"]
    EXP --> ISO["🚨 Isolation Forest\nAnomaly Detection"]
    EXP --> FC["📈 Forecasting\nNext Month Prediction"]
    EXP --> KM["👥 K-Means\nUser Clustering"]
    RF & SHAP & ISO & FC & KM --> CTX["📋 Context Builder"]
    CTX --> GEMINI["✨ Gemini 2.5 Flash\nFinancial Copilot"]
    GEMINI --> RESP["🗨 Personalized Advice"]
```

---

## ✨ Features

### 🔐 Authentication & Security

- **JWT Authentication** with configurable expiry (24h default)
- **Password Hashing** via bcrypt with salt rounds
- **Protected Routes** — all financial data requires a valid Bearer token
- **Token Expiry Detection** — client-side JWT expiry check without extra round-trips
- **Role-based Data Access** — users can only access their own financial records

### 💰 Financial Management

- **Expense Tracking** — CRUD for categorized expenses (food, housing, transport, etc.)
- **Budget Management** — set and monitor monthly budget limits per category
- **Financial Dashboard** — live score, category breakdown, trends at a glance
- **Spending Analytics** — monthly summaries, category totals, month-over-month trends
- **Personalized Recommendations** — ML-driven and AI-generated spending insights

### 🌲 Machine Learning

| Model | Purpose | Library |
|-------|---------|---------|
| **Random Forest** | Financial health score (0–100) | scikit-learn |
| **SHAP TreeExplainer** | Score factor attribution | shap |
| **K-Means Clustering** | User behavioral segmentation | scikit-learn |
| **Linear Trend / Holt-Winters** | Next-month spending forecast | numpy / statsmodels |
| **Isolation Forest** | Spending anomaly detection | scikit-learn |

### ✨ Generative AI Copilot

- **Gemini 2.5 Flash** integration with `system_instruction` + per-user data snapshot
- **Context-Aware Coaching** — every response references real score, income, spending, anomalies, and goals
- **Financial Goal Planning** — advice anchored to the user's stated objective
- **Indian Financial Context** — ₹-denominated advice, relevant to Indian market conditions
- **Graceful Fallback** — rule-based advisor activates if Gemini API is unavailable
- **Off-Topic Guard** — refuses non-finance questions with a polite redirect

---

## 🛠 Tech Stack

### Backend

| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.11+ | Core runtime |
| Flask | 3.1 | REST API framework |
| Flask-SQLAlchemy | 3.1 | ORM |
| Alembic | Latest | Database migrations |
| Flask-CORS | 6.0 | Cross-origin resource sharing |
| PyJWT | 2.8 | Token generation & verification |
| bcrypt | 4.1 | Password hashing |
| gunicorn | Latest | WSGI production server |

### Frontend

| Technology | Version | Role |
|-----------|---------|------|
| React | 19 | UI framework |
| TypeScript | 6.0 | Type safety |
| Vite | 8.0 | Build tool & dev server |
| Zustand | 5.0 | Lightweight state management |
| Axios | 1.17 | HTTP client with interceptors |
| Chart.js / react-chartjs-2 | 4.5 | Data visualizations |
| Framer Motion | 12 | Animations & transitions |
| React Router | 7 | Client-side routing |
| TanStack Query | 5 | Server state caching |

### Machine Learning

| Library | Role |
|--------|------|
| scikit-learn | Random Forest, K-Means, Isolation Forest |
| SHAP | Explainability & feature attribution |
| pandas | Data manipulation & feature engineering |
| numpy | Numerical computations |
| joblib | Model serialization |

### Data & Infrastructure

| Technology | Role |
|-----------|------|
| PostgreSQL (Supabase) | Production database |
| SQLite | Local development fallback |
| Google Gemini 2.5 Flash | Generative AI copilot |
| AWS | Production hosting target |
| python-dotenv | Environment configuration |

---

## 📁 Project Structure

```
nexus-finance/
│
├── backend/                             # Flask REST API
│   ├── routes/                          # Blueprint route handlers
│   │   ├── auth.py                      # /auth/* — register, login, me, logout
│   │   ├── expenses.py                  # /expenses/* — CRUD
│   │   ├── budget.py                    # /budget/* — budget management
│   │   ├── ml.py                        # /predict, /insights, /cluster_users
│   │   ├── analytics.py                 # /summary, /anomalies, /forecast
│   │   ├── recommendations.py           # /recommendations, /tips
│   │   ├── copilot.py                   # /copilot/chat — Gemini copilot
│   │   └── health.py                    # /health — liveness check
│   │
│   ├── services/                        # Business logic layer
│   │   ├── ml_service.py                # Random Forest score & feature engineering
│   │   ├── shap_service.py              # SHAP explainability
│   │   ├── forecasting_service.py       # Spending forecasting
│   │   ├── anomaly_service.py           # Isolation Forest detection
│   │   ├── context_service.py           # Gemini context builder
│   │   ├── gemini_service.py            # Gemini 2.5 Flash integration
│   │   └── fallback_advisor_service.py  # Rule-based fallback
│   │
│   ├── alembic/                         # Database migrations
│   │   ├── env.py                       # Migration environment
│   │   └── versions/                    # Migration scripts
│   │
│   ├── tests/
│   │   └── test_phase5.py               # Integration & unit tests
│   │
│   ├── app.py                           # Flask app factory
│   ├── auth.py                          # JWT helpers & decorators
│   ├── models.py                        # SQLAlchemy models
│   ├── config.py                        # Environment-based configuration
│   ├── finance_model.pkl                # Pre-trained Random Forest model
│   ├── alembic.ini                      # Alembic configuration
│   └── requirements.txt                 # Python dependencies
│
├── frontend/                            # React TypeScript SPA
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx          # Marketing / hero page
│   │   │   ├── AuthPage.tsx             # Login & registration
│   │   │   └── DashboardPage.tsx        # Main application dashboard
│   │   │
│   │   ├── services/
│   │   │   └── api.ts                   # Axios instance + JWT interceptors
│   │   │
│   │   ├── store/
│   │   │   ├── useAuthStore.ts          # Auth state (Zustand)
│   │   │   ├── useBudgetStore.ts        # Budget state (Zustand)
│   │   │   └── useDemoStore.ts          # Demo mode state
│   │   │
│   │   ├── styles/                      # CSS modules & global styles
│   │   └── App.tsx                      # Root component & routing
│   │
│   ├── vite.config.ts                   # Vite build configuration
│   ├── tsconfig.json                    # TypeScript configuration
│   └── package.json                     # Node dependencies
│
├── .env.example                         # Environment variable template
└── README.md
```

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | ❌ | Register new user |
| `POST` | `/auth/login` | ❌ | Authenticate and receive JWT token |
| `GET` | `/auth/me` | ✅ | Get current user profile |
| `POST` | `/auth/logout` | ✅ | Invalidate session |

### Expenses

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/expenses` | ✅ | List all user expenses |
| `POST` | `/expenses` | ✅ | Create new expense record |
| `PUT` | `/expenses/:id` | ✅ | Update expense |
| `DELETE` | `/expenses/:id` | ✅ | Delete expense |

### Machine Learning & Analytics

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/predict` | ✅ | Calculate financial health score (0–100) |
| `GET` | `/insights` | ✅ | SHAP-based score factor analysis |
| `GET` | `/summary` | ✅ | Monthly spending summary |
| `GET` | `/anomalies` | ✅ | Detected unusual transactions |
| `GET` | `/forecast` | ✅ | Next-month spending forecast |
| `GET` | `/recommendations` | ✅ | Personalized financial recommendations |
| `GET` | `/cluster_users` | ✅ | User behavioral cluster info |

### Budget & Portfolio

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/budget` | ✅ | Get budget limits & utilization |
| `PUT` | `/budget` | ✅ | Update budget configuration |
| `GET` | `/portfolio/valuation` | ✅ | Investment portfolio summary |

### Copilot & Utilities

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/copilot/chat` | ✅ | Send message to Gemini financial copilot |
| `GET` | `/health` | ❌ | API liveness check |

---

## 🔐 Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as React Frontend
    participant B as Flask API
    participant DB as PostgreSQL

    U->>F: Enter email + password
    F->>B: POST /auth/login {email, password}
    B->>DB: Query user by email
    DB-->>B: User record
    B->>B: bcrypt.verify(password, hash)
    B->>B: jwt.encode({user_id, exp})
    B-->>F: {token, user}
    F->>F: localStorage.setItem jwt_token
    F->>F: Zustand isAuthenticated = true
    F-->>U: Redirect to /dashboard

    Note over F,B: All subsequent requests
    F->>B: GET /auth/me Authorization Bearer token
    B->>B: jwt.decode(token, secret)
    B->>DB: User.query.get(user_id)
    B-->>F: {user: {...}}
```

---

## 📊 Financial Health Score Pipeline

```mermaid
flowchart TD
    A["💳 User Expense Data"] --> B["Feature Engineering"]
    B --> C{"Features"}
    C --> D["monthly_income"]
    C --> E["monthly_expense_total"]
    C --> F["investment_amount"]
    C --> G["savings_rate"]
    C --> H["expense_to_income_ratio"]
    D & E & F & G & H --> I["🌲 Random Forest Model\nfinance_model.pkl"]
    I --> J["Raw Score Prediction"]
    J --> K["Score Clipping 0-100"]
    K --> L["Financial Health Score"]
    L --> M["SHAP TreeExplainer"]
    M --> N["Feature SHAP Values"]
    N --> O["Split: Positive / Negative Drivers"]
    O --> P["Top Positive Factor\ne.g. Savings Rate"]
    O --> Q["Top Negative Factor\ne.g. High Expenses"]
```

The model is trained on a synthetic-but-realistic financial dataset and serialized as `finance_model.pkl`. It accepts three primary inputs — monthly income, total expenses, and investment amount — with engineered ratios computed at inference time.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) makes the Random Forest's decisions interpretable. Unlike black-box scores, Nexus Finance tells users **exactly which financial behaviors** are helping or hurting their score.

```mermaid
flowchart LR
    MODEL["Trained Random Forest"] --> EXPLAINER["SHAP TreeExplainer"]
    FEATURES["User Feature Vector"] --> EXPLAINER
    EXPLAINER --> VALUES["SHAP Value Array\nper Feature"]
    VALUES --> POSITIVE["Positive Drivers\nImpact greater than 0"]
    VALUES --> NEGATIVE["Negative Drivers\nImpact less than 0"]
    POSITIVE --> UI1["Your savings rate\nis boosting your score"]
    NEGATIVE --> UI2["High food expenses\nare reducing your score"]
```

**Example SHAP Output:**

```json
{
  "positive": [
    { "label": "Savings Rate",        "impact": 12.4 },
    { "label": "Investment Amount",   "impact":  6.1 }
  ],
  "negative": [
    { "label": "Total Expenses",           "impact": -9.2 },
    { "label": "Expense-to-Income Ratio",  "impact": -5.8 }
  ]
}
```

These factors are injected verbatim into the **Gemini Copilot context**, enabling the AI to reference real numbers when giving personalized advice.

---

## 📈 Spending Forecasting Pipeline

```mermaid
flowchart LR
    HIST["📅 Historical\nMonthly Totals"] --> CHECK{"Enough Data?\n2+ months"}
    CHECK -->|No| NO_DATA["Return has_data: false"]
    CHECK -->|Yes| TREND["Calculate Trend\nLinear Regression"]
    TREND --> FORECAST["Next Month Estimate"]
    FORECAST --> MSG{"Trend Direction"}
    MSG -->|Increasing| WARN["Spending trending upward\nReview your budget"]
    MSG -->|Decreasing| GOOD["Great work\nSpending is decreasing"]
    MSG -->|Stable| NEUTRAL["Spending is stable"]
    WARN & GOOD & NEUTRAL --> RESP["Forecast Response\nTrend + Message + Value"]
```

---

## 🚨 Anomaly Detection

Powered by **Isolation Forest** — an unsupervised algorithm that identifies transactions statistically unlike a user's normal spending patterns.

```mermaid
flowchart TD
    EXP["All User Expenses"] --> CHECK{"Has Sufficient\nHistory?"}
    CHECK -->|"Less than 10 records"| SKIP["Return has_data: false"]
    CHECK -->|"10+ records"| FEATURES["Extract Features\namount / day_of_month / category_encoded"]
    FEATURES --> ISO["Isolation Forest\ncontamination=0.1"]
    ISO --> PRED["Anomaly Scores"]
    PRED --> FILTER["Flag Records\nwhere score == -1"]
    FILTER --> RANK["Rank by Anomaly Severity"]
    RANK --> TOP["Return Top Anomalies\nwith Reason Strings"]
    TOP --> LOG["Persist to AnomalyLog Table"]
```

Anomaly reasons surface as human-readable strings:
- *"Unusually high amount for food (₹4,200 vs avg ₹850)"*
- *"First-time transaction in healthcare category"*

---

## ✨ Gemini 2.5 Flash Copilot

The copilot is not a generic chatbot — it is a **financially-grounded advisor** that sees the user's real data before every response.

### Request Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as React Frontend
    participant API as Flask /copilot/chat
    participant CTX as Context Builder
    participant GEM as Gemini 2.5 Flash
    participant FB as Fallback Advisor

    U->>F: How can I improve my score?
    F->>API: POST /copilot/chat {message}
    API->>CTX: build_financial_context(user)
    CTX->>CTX: Score + SHAP + Forecast + Anomalies\n+ Budget Utilization + MoM Change
    CTX-->>API: Context Dict (14 fields)
    API->>GEM: system_instruction + DATA_BLOCK + USER_QUESTION

    alt Gemini Available
        GEM-->>API: Structured advice 120-220 words
        API-->>F: {reply, mode: "gemini"}
    else API Error or Timeout
        API->>FB: ask_fallback_advisor()
        FB-->>API: Rule-based response
        API-->>F: {reply, mode: "fallback"}
    end

    F-->>U: Display formatted response
```

### Context Fields Injected Per Request

| Field | Description |
|-------|-------------|
| `score` | Financial health score (0–100) |
| `income` | Monthly income |
| `current_month_total` | Total spent this month |
| `budget_utilization_pct` | % of income spent |
| `savings_rate` | % of income saved |
| `mom_change_pct` | Month-over-month spending change |
| `top_category` + `top_category_amount` | Highest spend category |
| `active_categories` | Distinct expense categories this month |
| `top_positive_label` | Top SHAP positive factor |
| `top_negative_label` | Top SHAP negative factor |
| `forecast_trend` + `forecast_message` | Spending outlook |
| `anomaly_count` + `anomaly_summaries` | Detected unusual transactions |
| `goal` | User's stated financial goal |
| `risk_profile` | low / moderate / high |

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Tanmayy-k/nexus-finance.git
cd nexus-finance
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env         # Windows
cp .env.example .env           # macOS / Linux
# Edit .env — see Environment Variables section

# Run database migrations
python -m alembic upgrade head

# Start the API server
python app.py
# API available at: http://127.0.0.1:5000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "VITE_API_BASE_URL=http://127.0.0.1:5000" > .env

# Start development server
npm run dev
# App available at: http://localhost:5173
```

---

## 🔑 Environment Variables

### `backend/.env`

```env
# ── Application ───────────────────────────────────────────────────────────────
FLASK_ENV=development
SECRET_KEY=your-flask-secret-key-minimum-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-minimum-32-chars

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
SUPABASE_DB_POOL_SIZE=5
SUPABASE_DB_MAX_OVERFLOW=10

# ── AI ─────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY=your-gemini-api-key-from-google-ai-studio

# ── Feature Flags ──────────────────────────────────────────────────────────────
ALLOW_SEED=false
```

### `frontend/.env`

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

> **Security Note:** Never commit `.env` files. A `.env.example` template with no values is included in the repository.

---

## 🗄 PostgreSQL / Supabase Setup

Nexus Finance uses **Supabase** as its managed PostgreSQL provider. SQLite is the zero-config local fallback — no additional setup required for development.

### Supabase Configuration

1. Sign up at [supabase.com](https://supabase.com) and create a new project
2. Navigate to **Project Settings → Database → Connection String → URI**
3. Copy the URI and replace `[PASSWORD]` with your database password
4. Set the result as `DATABASE_URL` in `backend/.env`

### Running Migrations

```bash
cd backend

# Generate migration from current models
python -m alembic revision --autogenerate -m "initial_schema"

# Apply all pending migrations
python -m alembic upgrade head

# Rollback one migration
python -m alembic downgrade -1
```

### Database Schema

| Table | Purpose |
|-------|---------|
| `user` | User accounts, profile, income, financial goals |
| `expense` | Categorized expense records |
| `prediction_log` | ML score history with SHAP values (JSONB) |
| `anomaly_log` | Detected anomalies with reason descriptions |

---

## ☁️ AWS Deployment

Nexus Finance is architected for AWS deployment with environment-based configuration and production-grade PostgreSQL connection pooling.

### Recommended Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          AWS Cloud                           │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Route 53 │───▶│  CloudFront  │───▶│    S3 Bucket      │  │
│  │  (DNS)   │    │  (CDN+HTTPS) │    │   React Build     │  │
│  └──────────┘    └──────────────┘    └───────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    EC2 / ECS                          │  │
│  │   gunicorn --workers 4 --bind 0.0.0.0:8000 app:app   │  │
│  │   Flask REST API + ML Services + Gemini Copilot       │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │    Supabase     │                        │
│                   │   PostgreSQL    │                        │
│                   └─────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Backend Deployment (EC2 / ECS)

```bash
pip install -r requirements.txt

export FLASK_ENV=production
export DATABASE_URL=postgresql://...
export GEMINI_API_KEY=...
export JWT_SECRET_KEY=...

python -m alembic upgrade head
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

### Frontend Deployment (S3 + CloudFront)

```bash
cd frontend
VITE_API_BASE_URL=https://api.your-domain.com npm run build

aws s3 sync dist/ s3://your-bucket-name --delete
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

### Deployment Checklist

- [ ] `DATABASE_URL` points to production PostgreSQL (Supabase)
- [ ] `JWT_SECRET_KEY` is a cryptographically strong random string (min 64 chars)
- [ ] `GEMINI_API_KEY` is set and valid
- [ ] `FLASK_ENV=production` is set
- [ ] `ALLOW_SEED=false` in production
- [ ] Alembic migrations applied: `alembic upgrade head`
- [ ] CORS origins include the production frontend domain
- [ ] gunicorn running with 2+ workers
- [ ] HTTPS enforced on all API endpoints
- [ ] S3 bucket versioning enabled
- [ ] CloudFront HTTPS-only policy set
- [ ] Secrets stored in AWS Secrets Manager or Parameter Store

---

## 🔮 Future Roadmap

| Feature | Priority | Status |
|---------|----------|--------|
| 📱 React Native Mobile App | High | Planned |
| 🔔 Real-time Budget Alerts via WebSockets | High | Planned |
| 📊 Investment Portfolio Tracker with live prices | High | Planned |
| 🏦 Bank Account Integration (Plaid / Finbox) | Medium | Planned |
| 🤖 AI-Generated Monthly Financial Report (PDF) | Medium | Planned |
| 💬 Multi-turn Conversation Memory for Copilot | Medium | Planned |
| 🔐 2FA & OAuth (Google, GitHub) | Medium | Planned |
| 📤 CSV / PDF Data Export | Low | Planned |
| 🌐 Multi-currency Support | Low | Planned |
| 🧠 Fine-tuned Finance Domain LLM | Research | Exploring |

---

## 📚 Learning Outcomes

Building Nexus Finance involved solving production-grade engineering challenges across the full stack.

**Machine Learning in Production**
- Training, evaluating, and serializing a Random Forest model with joblib
- Implementing SHAP TreeExplainer for post-hoc model interpretability
- Isolation Forest contamination tuning for financial anomaly detection
- Feature engineering pipelines from raw transaction data

**Generative AI Engineering**
- Designing structured prompts that inject tabular data, not just free text
- Separating `system_instruction` from user-turn context in Gemini API
- Building graceful fallbacks for third-party API failures
- Managing context window budget for multi-field financial snapshots

**Backend Architecture**
- Flask application factory pattern with Blueprint-based route organization
- SQLAlchemy ORM with PostgreSQL-compatible JSONB columns (with SQLite fallback)
- Alembic migration management across SQLite development → PostgreSQL production
- Connection pooling for Supabase (pgBouncer-compatible settings)
- JWT authentication middleware with client-side and server-side expiry handling

**Frontend Engineering**
- React 19 with TypeScript in strict mode
- Zustand for minimal-boilerplate global state management
- Axios interceptors for automatic token injection and 401 session expiry handling
- Demo mode with mock API response interception (no backend required for demos)

**DevOps & Infrastructure**
- PostgreSQL dialect normalization (`postgres://` → `postgresql://` auto-fix)
- Environment-based configuration switching (SQLite dev → PostgreSQL prod)
- CORS policy management for multi-origin frontend deployments
- AWS S3 + CloudFront static hosting and cache invalidation pipeline

---

## 👨‍💻 Author

<div align="center">

### Tanmay Kshirsagar 

*Full-Stack Developer · ML Engineer · Fintech Builder*

[![GitHub](https://img.shields.io/badge/GitHub-Tanmayy--k-181717?style=for-the-badge&logo=github)](https://github.com/Tanmayy-k)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)]([https://linkedin.com/in/your-profil](https://www.linkedin.com/in/tanmay-kshirsagar-8188042b2/)

</div>

---

<div align="center">

**Python · Flask · React · TypeScript · Scikit-learn · SHAP · Google Gemini · PostgreSQL · AWS**

*A production-grade portfolio project demonstrating full-stack ML engineering,*
*Explainable AI, Generative AI integration, and cloud-ready architecture.*

---

⭐ **If this project impressed you, please give it a star — it helps others discover it.**

</div>
