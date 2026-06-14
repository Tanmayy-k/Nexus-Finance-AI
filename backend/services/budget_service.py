"""
Budget allocation service.

Implemented split (NOT 50/30/20):
  Housing 30%, Food 15%, Transportation 10%, Utilities 10%,
  Entertainment 5%, Savings 30%  (total allocated: 100%)
"""

import logging

logger = logging.getLogger(__name__)


def normalize_category(cat: str) -> str:
    return cat.lower().strip() if cat else ''


def calculate_budget_split(income: float) -> dict:
    """Return category budget allocations for a given monthly income."""
    income = float(income)
    return {
        "Housing": round(income * 0.30, 2),
        "Food": round(income * 0.15, 2),
        "Transportation": round(income * 0.10, 2),
        "Utilities": round(income * 0.10, 2),
        "Entertainment": round(income * 0.05, 2),
        "Savings": round(income * 0.30, 2),
    }
