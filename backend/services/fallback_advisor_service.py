import logging
import re

logger = logging.getLogger(__name__)

OFF_TOPIC_KEYWORDS = (
    "politics", "election", "president", "weather", "python code",
    "javascript", "write code", "recipe", "movie", "sports score",
    "football", "cricket match", "who won", "tell me a joke",
    "capital of", "history of", "translate", "poem",
)

REFUSAL = "I can only help with your personal finances."


def ask_fallback_advisor(user_message: str, context: dict) -> str:
    """Rule-based financial advisor when Gemini is unavailable."""
    msg = user_message.strip().lower()
    if not msg:
        return "Please type a question."

    if _is_off_topic(msg):
        return REFUSAL

    if any(k in msg for k in ("score", "improve", "financial health", "better")):
        return _score_advice(context)

    if any(k in msg for k in ("forecast", "increasing", "decreasing", "next month", "spending trend")):
        return _forecast_advice(context)

    if any(k in msg for k in ("category", "categories", "hurts", "highest spend", "biggest expense")):
        return _category_advice(context)

    if any(k in msg for k in ("unusual", "anomaly", "flagged", "strange", "transaction")):
        return _anomaly_advice(context)

    if any(k in msg for k in ("save", "saving", "savings rate", "lakh", "crore", "goal")):
        savings_goal = _savings_goal_advice(msg, context)
        if savings_goal:
            return savings_goal
        return _savings_rate_advice(context)

    return _general_advice(context)


def _is_off_topic(msg: str) -> bool:
    finance_hints = (
        "money", "finance", "budget", "expense", "income", "save",
        "invest", "score", "forecast", "spend", "rupee", "₹",
    )
    if any(h in msg for h in finance_hints):
        return False
    return any(k in msg for k in OFF_TOPIC_KEYWORDS)


def _score_advice(context: dict) -> str:
    score = context.get("score")
    if score is None:
        return "Add income and expenses to calculate your financial score first."

    parts = [f"Your financial score is {score}/100."]
    if context.get("top_positive_label"):
        parts.append(f"Keep strengthening {context['top_positive_label']}.")
    if context.get("top_negative_label"):
        parts.append(
            f"Focus on improving {context['top_negative_label']} — "
            f"it is pulling your score down."
        )
    if context.get("savings_rate") is not None and context["savings_rate"] < 20:
        parts.append("Try to raise your savings rate above 20% of income.")
    return " ".join(parts)


def _forecast_advice(context: dict) -> str:
    if not context.get("forecast_message"):
        return "Track expenses for at least 2 months to unlock spending forecasts."

    trend = context.get("forecast_trend", "unknown")
    if trend == "increasing":
        return (
            f"{context['forecast_message']} "
            f"Review discretionary spending in {context.get('top_category') or 'your top categories'} "
            f"to slow the rise."
        )
    if trend == "decreasing":
        return (
            f"{context['forecast_message']} "
            "Good trend — maintain current discipline."
        )
    return context["forecast_message"]


def _category_advice(context: dict) -> str:
    if not context.get("top_category"):
        return "No expense data this month yet. Add transactions to see category breakdown."

    amt = context.get("top_category_amount", 0)
    return (
        f"{context['top_category']} is your highest spend this month at ₹{amt:,.0f}. "
        f"Cutting it by 10% would free up ₹{amt * 0.1:,.0f} monthly."
    )


def _anomaly_advice(context: dict) -> str:
    if context.get("anomaly_count", 0) == 0:
        return "No unusual transactions detected in the last 90 days."

    reasons = "; ".join(context.get("anomaly_summaries") or [])
    return (
        f"{context['anomaly_count']} unusual expense(s) found. "
        f"{reasons} Review these and confirm they were intentional."
    )


def _savings_goal_advice(msg: str, context: dict) -> str | None:
  # Match patterns like "10 lakh", "₹10 lakh", "10 lakh in 5 years"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore)?(?:\s+in\s+(\d+)\s*years?)?",
        msg,
    )
    if not match:
        return None

    amount = float(match.group(1))
    unit = (match.group(2) or "lakh").lower()
    years = int(match.group(3)) if match.group(3) else 5

    if unit == "crore":
        target = amount * 10_000_000
    else:
        target = amount * 100_000

    months = years * 12
    monthly = target / months
    return (
        f"To reach ₹{target:,.0f} in {years} years, save about "
        f"₹{monthly:,.0f} per month (ignoring interest). "
        f"Your current savings rate is {context.get('savings_rate', 'unknown')}%."
    )


def _savings_rate_advice(context: dict) -> str:
    rate = context.get("savings_rate")
    income = context.get("income") or 0
    if rate is None or not income:
        return "Set your income and track expenses to measure your savings rate."

    if rate < 20:
        return (
            f"Your savings rate is {rate}% — below the 20% target. "
            f"Reduce spending in {context.get('top_category') or 'discretionary categories'} "
            f"or increase investments."
        )
    return (
        f"Strong savings rate at {rate}%. "
        f"Consider allocating more to investments aligned with your "
        f"{context.get('risk_profile', 'moderate')} risk profile."
    )


def _general_advice(context: dict) -> str:
    parts = []
    if context.get("score") is not None:
        parts.append(f"Score: {context['score']}/100.")
    if context.get("forecast_message"):
        parts.append(context["forecast_message"])
    if context.get("top_category"):
        parts.append(
            f"Top spend: {context['top_category']} "
            f"(₹{context['top_category_amount']:,.0f})."
        )
    if not parts:
        return "Add expenses and income data to get personalized advice."
    return " ".join(parts)
