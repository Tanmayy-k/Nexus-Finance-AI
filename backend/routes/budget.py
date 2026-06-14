import logging
import random

from flask import Blueprint, request, jsonify

from auth import token_required
from models import db, Expense
from services.budget_service import calculate_budget_split

logger = logging.getLogger(__name__)

budget_bp = Blueprint('budget', __name__)


def _budget_payload(user):
    if not user.income:
        return None
    return {
        "income": user.income,
        "goal": user.goal,
        "budget_split": calculate_budget_split(user.income),
    }


@budget_bp.route("/budget", methods=["GET"])
@token_required
def get_budget(current_user):
    payload = _budget_payload(current_user)
    if payload is None:
        return jsonify({
            "has_budget": False,
            "message": "No budget set. Create one to get started.",
        }), 200
    return jsonify({"has_budget": True, **payload}), 200


@budget_bp.route("/budget", methods=["POST"])
@token_required
def budget(current_user):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    income = data.get("income")
    goal = data.get("goal")
    risk_profile = data.get("risk_profile")

    if not income:
        return jsonify({"error": "Income is required"}), 400

    try:
        current_user.income = float(income)
        current_user.goal = goal
        current_user.risk_profile = risk_profile
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Budget update failed: %s", e)
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    budget_split = calculate_budget_split(current_user.income)

    return jsonify({"has_budget": True, **{
        "income": current_user.income,
        "goal": current_user.goal,
        "budget_split": budget_split,
    }})


@budget_bp.route("/tips", methods=["GET"])
@token_required
def tips(current_user):
    try:
        expenses = Expense.query.filter_by(user_id=current_user.id).all()
        expense_data = {}
        for e in expenses:
            key = e.category.lower()
            expense_data[key] = expense_data.get(key, 0) + e.amount

        if expense_data.get("food", 0) > 5000:
            return jsonify({"tip": "Try cooking at home instead of eating out to save more."})
        if expense_data.get("entertainment", 0) > 3000:
            return jsonify({
                "tip": "Look for free or low-cost entertainment options like a picnic or a library trip."
            })
        if expense_data.get("shopping", 0) > 7000:
            return jsonify({
                "tip": "Before you buy, ask yourself if you really need it. Pause and think."
            })

        tips_list = [
            "Track your spending daily to cut unnecessary costs.",
            "Start an SIP with even ₹500 — consistency matters.",
            "Maintain an emergency fund of 6 months' expenses.",
            "Avoid using credit cards for wants, stick to needs.",
            "Review and rebalance your budget every month.",
        ]
        return jsonify({"tip": random.choice(tips_list)})
    except Exception as e:
        logger.error("Tips fetch failed: %s", e)
        return jsonify({"error": f"Error fetching tips: {str(e)}"}), 500
