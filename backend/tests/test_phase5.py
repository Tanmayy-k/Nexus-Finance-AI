"""
Phase 5 verification tests.
Run with: cd backend && python -m pytest tests/test_phase5.py -v
Requires: backend running + valid .env with DATABASE_URL and GEMINI_API_KEY
"""
import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── A. PostgreSQL connection ──────────────────────────────────────────────────

def test_database_url_is_postgres():
    from config import Config
    url = Config.DATABASE_URL
    assert url.startswith('postgresql://'), (
        f"DATABASE_URL should start with 'postgresql://', got: {url[:30]}")

def test_sqlalchemy_can_connect():
    from app import create_app
    app = create_app()
    with app.app_context():
        from models import db
        result = db.session.execute(db.text('SELECT 1')).fetchone()
        assert result[0] == 1, "DB connection failed"

def test_all_tables_exist():
    from app import create_app
    app = create_app()
    with app.app_context():
        from models import db
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        required = ['user', 'expense', 'prediction_log', 'anomaly_log']
        for t in required:
            assert t in tables, f"Table '{t}' missing from database"

def test_connection_pool_configured():
    from config import Config
    opts = Config.SQLALCHEMY_ENGINE_OPTIONS
    assert 'pool_size' in opts, "pool_size not in engine options"
    assert 'pool_pre_ping' in opts, "pool_pre_ping not in engine options"
    assert opts['pool_pre_ping'] is True

# ── B. Models ─────────────────────────────────────────────────────────────────

def test_user_model_crud():
    from app import create_app
    app = create_app()
    with app.app_context():
        from models import db, User
        u = User(email='phase5test@test.com', name='Phase5',
                 income=60000.0)
        u.password = 'hashed'
        db.session.add(u)
        db.session.commit()
        fetched = User.query.filter_by(email='phase5test@test.com').first()
        assert fetched is not None
        assert fetched.income == 60000.0
        db.session.delete(fetched)
        db.session.commit()

def test_expense_model_crud():
    from app import create_app
    app = create_app()
    with app.app_context():
        from models import db, User, Expense
        from datetime import date
        u = User(email='exptest5@test.com', name='ExpTest', income=50000.0)
        u.password = 'hashed'
        db.session.add(u)
        db.session.flush()
        e = Expense(user_id=u.id, category='food',
                    amount=500.0, created_at=date.today())
        db.session.add(e)
        db.session.commit()
        fetched = Expense.query.filter_by(user_id=u.id).first()
        assert fetched.amount == 500.0
        assert fetched.category == 'food'
        db.session.delete(fetched)
        db.session.delete(u)
        db.session.commit()

# ── C. ML services ────────────────────────────────────────────────────────────

def test_shap_returns_factors():
    from app import create_app
    app = create_app()
    with app.app_context():
        from services.ml_service import build_feature_df, predict_score, get_model
        from services.shap_service import explain_score, split_factors
        df = build_feature_df(60000, 25000, 6000)
        score = predict_score(df)
        assert 0 <= score <= 100, f"Score out of range: {score}"
        model = get_model()
        factors = explain_score(model, df)
        assert isinstance(factors, list)
        split = split_factors(factors)
        assert 'positive' in split
        assert 'negative' in split

def test_forecasting_no_crash_on_empty():
    from app import create_app
    app = create_app()
    with app.app_context():
        from services.forecasting_service import forecast_next_month
        result = forecast_next_month(user_id=999999)
        assert result['has_data'] is False
        assert 'message' in result

def test_anomaly_no_crash_on_empty():
    from app import create_app
    app = create_app()
    with app.app_context():
        from services.anomaly_service import detect_anomalies
        result = detect_anomalies(user_id=999999)
        assert result['has_data'] is False
        assert 'anomalies' in result

# ── D. Context service ────────────────────────────────────────────────────────

def test_context_builds_without_crash():
    from app import create_app
    app = create_app()
    with app.app_context():
        from models import User
        from services.context_service import build_financial_context
        u = User(id=999999, name='Test', email='ctx@test.com',
                 income=50000.0, goal='Save', risk_profile='moderate')
        context = build_financial_context(u)
        assert 'score' in context
        assert 'user_name' in context
        assert 'forecast_trend' in context
        assert 'anomaly_count' in context

# ── E. Gemini service ─────────────────────────────────────────────────────────

def test_gemini_returns_non_empty_reply():
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set — skipping Gemini test")
    from services.gemini_service import ask_copilot
    context = {
        'user_name': 'TestUser', 'income': 60000, 'score': 55.0,
        'top_positive_label': 'Savings Rate',
        'top_negative_label': 'Total Expenses',
        'forecast_trend': 'stable', 'forecast_message': 'Spending is stable.',
        'anomaly_count': 0, 'anomaly_summaries': [],
        'top_category': 'Food', 'top_category_amount': 8000,
        'current_month_total': 30000, 'savings_rate': 40.0,
        'budget_utilization_pct': 50.0, 'mom_change_pct': -5.0,
        'goal': 'Save ₹5 lakh', 'risk_profile': 'moderate',
        'active_categories': 4,
    }
    reply = ask_copilot("How can I improve my financial score?", context)
    assert isinstance(reply, str)
    assert len(reply) > 50, f"Reply too short: {len(reply)} chars"
    assert not reply.endswith('...'), "Reply appears truncated"

def test_gemini_refuses_off_topic():
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set — skipping Gemini test")
    from services.gemini_service import ask_copilot
    context = {
        'user_name': 'TestUser', 'income': 60000, 'score': 55.0,
        'top_positive_label': None, 'top_negative_label': None,
        'forecast_trend': None, 'forecast_message': None,
        'anomaly_count': 0, 'anomaly_summaries': [],
        'top_category': None, 'top_category_amount': None,
        'current_month_total': 0, 'savings_rate': None,
        'budget_utilization_pct': None, 'mom_change_pct': None,
        'goal': 'Not specified', 'risk_profile': 'moderate',
        'active_categories': 0,
    }
    reply = ask_copilot("Who won the cricket World Cup?", context)
    finance_keywords = ['finance', 'budget', 'spending', 'saving', 'money']
    assert any(k in reply.lower() for k in finance_keywords), (
        f"Off-topic guard may not be working. Reply: {reply}")

# ── F. API endpoints (requires running server) ────────────────────────────────

def test_health_endpoint():
    import requests
    try:
        r = requests.get('http://localhost:5000/health', timeout=5)
        assert r.status_code == 200
        assert r.json().get('status') == 'ok'
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running — skipping endpoint test")

def test_copilot_endpoint_requires_auth():
    import requests
    try:
        r = requests.post('http://localhost:5000/copilot/chat',
                          json={'message': 'test'}, timeout=5)
        assert r.status_code == 401, "Copilot must require authentication"
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running — skipping endpoint test")
