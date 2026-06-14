import logging
from datetime import date

from sqlalchemy import func

from models import Expense, User, db

logger = logging.getLogger(__name__)


def build_financial_context(user: User) -> dict:
    """
    Aggregate all available financial data for a user into
    a single context dict for the Gemini prompt.

    Never raises — missing data fills with safe defaults.
    """
    context = {
        "user_name": user.name or "User",
        "income": user.income or 0,
        "goal": user.goal or "Not specified",
        "risk_profile": user.risk_profile or "moderate",
        "mom_change_pct": None,
        "active_categories": 0,
        "budget_utilization_pct": None,
        "score": None,
        "shap_positive": [],
        "shap_negative": [],
        "top_positive_label": None,
        "top_negative_label": None,
        "forecast_trend": None,
        "forecast_message": None,
        "forecast_next_month": None,
        "anomaly_count": 0,
        "anomaly_summaries": [],
        "top_category": None,
        "top_category_amount": None,
        "current_month_total": None,
        "savings_rate": None,
    }

    try:
        from services.ml_service import build_feature_df, predict_score, get_model
        from services.shap_service import explain_score, split_factors

        first_day = date.today().replace(day=1)
        monthly_total = (
            db.session.query(func.sum(Expense.amount))
            .filter(
                Expense.user_id == user.id,
                Expense.created_at >= first_day,
            )
            .scalar() or 0.0
        )
        investment_amount = user.income * 0.1 if user.income else 0

        features_df = build_feature_df(
            monthly_income=user.income or 0,
            monthly_expense_total=float(monthly_total),
            investment_amount=investment_amount,
        )
        score = predict_score(features_df.iloc[0].to_dict())
        context["score"] = round(float(score), 1)
        context["current_month_total"] = round(float(monthly_total), 2)

        if user.income and user.income > 0:
            context["savings_rate"] = round(
                (user.income - float(monthly_total)) / user.income * 100, 1
            )

        model = get_model()
        factors = explain_score(model, features_df)
        split = split_factors(factors)
        context["shap_positive"] = split["positive"][:3]
        context["shap_negative"] = split["negative"][:3]
        if split["positive"]:
            context["top_positive_label"] = split["positive"][0]["label"]
        if split["negative"]:
            context["top_negative_label"] = split["negative"][0]["label"]

    except Exception as e:
        logger.warning("Context: score/SHAP failed — %s", e)

    try:
        from services.forecasting_service import forecast_next_month

        fc = forecast_next_month(user.id)
        if fc.get("has_data"):
            context["forecast_trend"] = fc["trend"]
            context["forecast_message"] = fc["message"]
            context["forecast_next_month"] = fc["next_month_forecast"]
    except Exception as e:
        logger.warning("Context: forecast failed — %s", e)

    try:
        from services.anomaly_service import detect_anomalies

        anom = detect_anomalies(user.id)
        anomalies = anom.get("anomalies", [])
        context["anomaly_count"] = len(anomalies)
        context["anomaly_summaries"] = [
            a["reason"] for a in anomalies[:2]
        ]
    except Exception as e:
        logger.warning("Context: anomaly failed — %s", e)

    try:
        first_day = date.today().replace(day=1)
        result = (
            db.session.query(
                Expense.category,
                func.sum(Expense.amount).label("total"),
            )
            .filter(
                Expense.user_id == user.id,
                Expense.created_at >= first_day,
            )
            .group_by(Expense.category)
            .order_by(func.sum(Expense.amount).desc())
            .first()
        )
        if result:
            context["top_category"] = result[0].title()
            context["top_category_amount"] = round(float(result[1]), 2)
    except Exception as e:
        logger.warning("Context: top category failed — %s", e)

    # mom_change_pct — compare this month vs last month total
    try:
        from datetime import date
        from dateutil.relativedelta import relativedelta
        today = date.today()
        first_this = today.replace(day=1)
        first_last = (today.replace(day=1) - relativedelta(months=1))
        last_end   = first_this

        this_total = db.session.query(func.sum(Expense.amount)).filter(
            Expense.user_id == user.id,
            Expense.created_at >= first_this
        ).scalar() or 0

        last_total = db.session.query(func.sum(Expense.amount)).filter(
            Expense.user_id == user.id,
            Expense.created_at >= first_last,
            Expense.created_at < last_end
        ).scalar() or 0

        if last_total > 0:
            context["mom_change_pct"] = round(
                (this_total - last_total) / last_total * 100, 1
            )
    except Exception as e:
        logger.warning(f"Context: MoM change failed — {e}")

    # active_categories — distinct categories used this month
    try:
        from datetime import date
        cat_count = db.session.query(func.count(
            func.distinct(Expense.category)
        )).filter(
            Expense.user_id == user.id,
            Expense.created_at >= date.today().replace(day=1)
        ).scalar() or 0
        context["active_categories"] = cat_count
    except Exception as e:
        logger.warning(f"Context: category count failed — {e}")

    # budget_utilization_pct
    try:
        if user.income and user.income > 0 and context["current_month_total"]:
            context["budget_utilization_pct"] = round(
                context["current_month_total"] / user.income * 100, 1
            )
    except Exception as e:
        logger.warning(f"Context: budget utilization failed — {e}")

    logger.info(
        "Context built for user %s — score=%s, anomalies=%s",
        user.id,
        context["score"],
        context["anomaly_count"],
    )
    return context
