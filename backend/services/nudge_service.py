import logging
from datetime import date

from models import User, Expense, db
from services.budget_service import normalize_category

logger = logging.getLogger(__name__)


def get_smart_nudge(user_id, category, amount):
    """Generates a smart nudge based on the expense just added."""
    try:
        category = normalize_category(category)
        user = User.query.get(user_id)
        if not user or not user.income:
            return None

        budget_limits = {
            "housing": user.income * 0.30,
            "food": user.income * 0.15,
            "transportation": user.income * 0.10,
            "utilities": user.income * 0.10,
            "entertainment": user.income * 0.05,
            "savings": user.income * 0.30,
        }

        budget_limit = budget_limits.get(category, 0)
        if budget_limit == 0:
            return None

        today = date.today()
        start_of_month = today.replace(day=1)
        expenses = Expense.query.filter(
            Expense.user_id == user_id,
            Expense.category == category,
            Expense.created_at >= start_of_month
        ).all()

        total_spent = sum(e.amount for e in expenses)

        if total_spent > budget_limit:
            return (
                f"Budget Alert! You've spent ₹{total_spent:,.0f} on '{category}' this month, "
                f"which is over your ₹{budget_limit:,.0f} budget."
            )
        if total_spent > budget_limit * 0.8:
            return (
                f"Heads up! You're at {int((total_spent / budget_limit) * 100)}% "
                f"of your '{category}' budget for the month."
            )

        if category == "food" and amount < 300:
            count = Expense.query.filter(
                Expense.user_id == user_id,
                Expense.category == "food",
                Expense.amount < 300
            ).count()
            if count > 10:
                return f"This is your {count}th small food purchase this month. These can add up!"

        return None
    except Exception as e:
        logger.error("Nudge error: %s", e)
        return None
