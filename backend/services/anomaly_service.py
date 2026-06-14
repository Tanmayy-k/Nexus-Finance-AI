import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from models import AnomalyLog, Expense, db

logger = logging.getLogger(__name__)

MIN_EXPENSES_REQUIRED = 10
LOOKBACK_DAYS = 90
CONTAMINATION = 0.1


def detect_anomalies(user_id: int) -> dict:
    """Detect anomalous expenses using Isolation Forest."""
    try:
        cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)
        expenses = Expense.query.filter(
            Expense.user_id == user_id,
            Expense.created_at >= cutoff,
        ).all()

        if len(expenses) < MIN_EXPENSES_REQUIRED:
            return {
                "has_data": False,
                "total_analyzed": len(expenses),
                "anomalies": [],
                "message": (
                    f"Need at least {MIN_EXPENSES_REQUIRED} expenses "
                    f"in the last {LOOKBACK_DAYS} days. "
                    f"Currently have {len(expenses)}."
                ),
            }

        df = pd.DataFrame([{
            "id": e.id,
            "category": e.category,
            "amount": e.amount,
            "day_of_week": _day_of_week(e.created_at),
            "description": e.description or "",
            "date": str(e.created_at),
        } for e in expenses])

        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["category"])

        features = df[["amount", "day_of_week", "category_encoded"]].values

        clf = IsolationForest(
            contamination=CONTAMINATION,
            random_state=42,
            n_estimators=100,
        )
        predictions = clf.fit_predict(features)
        scores = clf.decision_function(features)

        normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, normalized)):
            if pred == -1:
                row = df.iloc[i]
                anomalies.append({
                    "id": int(row["id"]),
                    "category": row["category"].title(),
                    "amount": float(row["amount"]),
                    "date": row["date"],
                    "description": row["description"],
                    "anomaly_score": round(float(score), 3),
                    "reason": _generate_reason(row, df),
                })

        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)

        try:
            for a in anomalies:
                existing = AnomalyLog.query.filter_by(
                    user_id=user_id,
                    expense_id=a["id"],
                ).first()
                if not existing:
                    log = AnomalyLog(
                        user_id=user_id,
                        expense_id=a["id"],
                        anomaly_score=a["anomaly_score"],
                        reason=a["reason"],
                    )
                    db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.warning("AnomalyLog insert failed (non-critical): %s", e)
            db.session.rollback()

        return {
            "has_data": True,
            "total_analyzed": len(expenses),
            "anomalies": anomalies,
            "message": (
                f"{len(anomalies)} unusual expense(s) detected "
                f"out of {len(expenses)} analyzed."
                if anomalies else
                "No unusual spending patterns detected."
            ),
        }

    except Exception as e:
        logger.error("Anomaly detection failed for user %s: %s", user_id, e)
        return {
            "has_data": False,
            "total_analyzed": 0,
            "anomalies": [],
            "message": "Anomaly detection unavailable.",
        }


def _day_of_week(dt) -> int:
    if hasattr(dt, 'weekday'):
        return dt.weekday()
    return 0


def _generate_reason(row: pd.Series, df: pd.DataFrame) -> str:
    """Generate a human-readable reason for why this expense is anomalous."""
    cat = row["category"]
    amount = row["amount"]

    cat_expenses = df[df["category"] == cat]["amount"]
    if len(cat_expenses) > 1:
        mean_amt = cat_expenses.mean()
        if amount > mean_amt * 2:
            return (
                f"{cat.title()} expense is "
                f"{amount / mean_amt:.1f}x higher than your usual "
                f"₹{mean_amt:,.0f} average."
            )
    return (
        f"Unusual {cat.title()} expense of ₹{amount:,.0f} "
        f"detected based on your spending pattern."
    )
