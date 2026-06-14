import logging
from datetime import date

from sqlalchemy import func

from models import Expense, User, db
from services.anomaly_service import detect_anomalies
from services.forecasting_service import forecast_next_month

logger = logging.getLogger(__name__)


def generate_insights(user_id: int) -> dict:
    """Generate actionable financial insights from forecast, anomalies, and spend data."""
    insights = []

    forecast = forecast_next_month(user_id)
    if forecast["has_data"]:
        sev = (
            "warning" if forecast["trend"] == "increasing"
            else "positive" if forecast["trend"] == "decreasing"
            else "info"
        )
        insights.append({
            "type": "forecast",
            "severity": sev,
            "message": forecast["message"],
        })

    anomaly_result = detect_anomalies(user_id)
    for anomaly in anomaly_result.get("anomalies", [])[:2]:
        insights.append({
            "type": "anomaly",
            "severity": "warning",
            "message": anomaly["reason"],
        })

    category_insight = _top_category_insight(user_id)
    if category_insight:
        insights.append(category_insight)

    savings_insight = _savings_insight(user_id)
    if savings_insight:
        insights.append(savings_insight)

    if not insights:
        insights.append({
            "type": "general",
            "severity": "info",
            "message": "Add more expenses to unlock personalized insights.",
        })

    summary = _build_summary(forecast, anomaly_result)

    return {"insights": insights, "summary": summary}


def _top_category_insight(user_id: int) -> dict | None:
    """Flag the highest spending category this month."""
    try:
        first_day = date.today().replace(day=1)
        results = (
            db.session.query(Expense.category, func.sum(Expense.amount).label("total"))
            .filter(Expense.user_id == user_id, Expense.created_at >= first_day)
            .group_by(Expense.category)
            .order_by(func.sum(Expense.amount).desc())
            .first()
        )
        if results:
            cat, total = results
            return {
                "type": "category",
                "severity": "info",
                "message": (
                    f"{cat.title()} is your highest spend category "
                    f"this month at ₹{total:,.0f}. "
                    f"Reducing it by 10% could improve your financial score."
                ),
            }
    except Exception as e:
        logger.warning("Category insight failed: %s", e)
    return None


def _savings_insight(user_id: int) -> dict | None:
    """Encourage saving if total expenses exceed 80% of income."""
    try:
        user = User.query.get(user_id)
        if not user or not user.income:
            return None

        first_day = date.today().replace(day=1)
        total_spent = (
            db.session.query(func.sum(Expense.amount))
            .filter(Expense.user_id == user_id, Expense.created_at >= first_day)
            .scalar() or 0
        )
        ratio = total_spent / user.income
        if ratio > 0.8:
            return {
                "type": "general",
                "severity": "warning",
                "message": (
                    f"You've spent {ratio * 100:.0f}% of your monthly income. "
                    f"Consider reviewing discretionary expenses."
                ),
            }
        if ratio < 0.5:
            return {
                "type": "general",
                "severity": "positive",
                "message": (
                    f"Great discipline — only {ratio * 100:.0f}% of income spent "
                    f"this month. You're on track to save well."
                ),
            }
    except Exception as e:
        logger.warning("Savings insight failed: %s", e)
    return None


def _build_summary(forecast: dict, anomaly_result: dict) -> str:
    parts = []
    if forecast.get("has_data"):
        parts.append(f"Forecast: {forecast['trend']}.")
    n = len(anomaly_result.get("anomalies", []))
    if n:
        parts.append(f"{n} anomaly detected.")
    return " ".join(parts) if parts else "Insufficient data for summary."
