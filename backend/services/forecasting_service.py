import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models import Expense

logger = logging.getLogger(__name__)

MIN_MONTHS_REQUIRED = 2


def forecast_next_month(user_id: int) -> dict:
    """
    Forecast next month's total spending using Linear Regression
    on the user's monthly expense history.
    """
    try:
        expenses = Expense.query.filter_by(user_id=user_id).all()

        if not expenses:
            return _no_data_response("No expense data found.")

        records = [
            {"month": _month_key(e.created_at), "amount": e.amount}
            for e in expenses
        ]
        df = pd.DataFrame(records)
        monthly = df.groupby("month")["amount"].sum().reset_index()
        monthly = monthly.sort_values("month").reset_index(drop=True)

        if len(monthly) < MIN_MONTHS_REQUIRED:
            return _no_data_response(
                f"Need at least {MIN_MONTHS_REQUIRED} months of data. "
                f"Currently have {len(monthly)}."
            )

        X = np.array(monthly.index).reshape(-1, 1)
        y = monthly["amount"].values
        model = LinearRegression()
        model.fit(X, y)

        next_index = np.array([[len(monthly)]])
        next_month_forecast = float(model.predict(next_index)[0])
        next_month_forecast = max(0.0, next_month_forecast)

        current_month_total = float(monthly["amount"].iloc[-1])

        if current_month_total > 0:
            change_pct = (
                (next_month_forecast - current_month_total) / current_month_total * 100
            )
        else:
            change_pct = 0.0

        if change_pct > 5:
            trend = "increasing"
        elif change_pct < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "has_data": True,
            "months_available": len(monthly),
            "current_month_total": round(current_month_total, 2),
            "next_month_forecast": round(next_month_forecast, 2),
            "change_pct": round(change_pct, 1),
            "trend": trend,
            "message": _trend_message(trend, change_pct, next_month_forecast),
        }

    except Exception as e:
        logger.error("Forecasting failed for user %s: %s", user_id, e)
        return _no_data_response("Forecasting unavailable.")


def _month_key(dt) -> str:
    """Convert a date/datetime to 'YYYY-MM' string."""
    if hasattr(dt, 'strftime'):
        return dt.strftime("%Y-%m")
    return str(dt)[:7]


def _trend_message(trend: str, change_pct: float, forecast: float) -> str:
    amt = f"₹{forecast:,.0f}"
    if trend == "increasing":
        return f"Spending projected to increase {abs(change_pct):.1f}% next month (~{amt})."
    if trend == "decreasing":
        return f"Spending projected to decrease {abs(change_pct):.1f}% next month (~{amt})."
    return f"Spending looks stable next month (~{amt})."


def _no_data_response(message: str) -> dict:
    return {
        "has_data": False,
        "months_available": 0,
        "current_month_total": 0.0,
        "next_month_forecast": 0.0,
        "change_pct": 0.0,
        "trend": "unknown",
        "message": message,
    }
