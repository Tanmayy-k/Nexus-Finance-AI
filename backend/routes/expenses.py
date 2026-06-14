import logging
from datetime import datetime, date

from flask import Blueprint, request, jsonify

from auth import token_required
from models import db, Expense
from services.budget_service import normalize_category
from services.nudge_service import get_smart_nudge

logger = logging.getLogger(__name__)

expenses_bp = Blueprint('expenses', __name__)


@expenses_bp.route("/expense", methods=["POST"])
@token_required
def expense(current_user):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    user_id = current_user.id
    category = normalize_category(data.get("category", ""))
    amount = data.get("amount")
    date_str = data.get("date")

    if not all([category, amount]):
        return jsonify({"error": "category and amount are required"}), 400

    try:
        if date_str:
            try:
                created_at = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return jsonify({"error": "date must be YYYY-MM-DD"}), 400
        else:
            created_at = date.today()

        exp = Expense(
            user_id=user_id,
            category=category,
            amount=float(amount),
            description=data.get("description"),
            created_at=created_at,
        )
        db.session.add(exp)
        db.session.commit()

        nudge_message = get_smart_nudge(user_id, category, float(amount))

        return jsonify({
            "message": "Expense added successfully",
            "category": category,
            "amount": float(amount),
            "date": created_at.isoformat(),
            "nudge_message": nudge_message,
        })
    except Exception as e:
        db.session.rollback()
        logger.error("Expense creation failed: %s", e)
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@expenses_bp.route("/expenses", methods=["GET"])
@token_required
def list_expenses(current_user):
    expenses = (
        Expense.query
        .filter_by(user_id=current_user.id)
        .order_by(Expense.created_at.desc())
        .all()
    )
    return jsonify({
        "expenses": [{
            "id": e.id,
            "category": e.category.title(),
            "amount": e.amount,
            "description": e.description or "",
            "date": e.created_at.isoformat() if e.created_at else None,
        } for e in expenses]
    })


@expenses_bp.route("/summary", methods=["GET"])
@token_required
def summary(current_user):
    try:
        expenses = Expense.query.filter_by(user_id=current_user.id).all()

        total_spent = sum(e.amount for e in expenses)
        expense_data = {}
        for e in expenses:
            key = e.category.lower()
            expense_data[key] = expense_data.get(key, 0) + e.amount

        by_category = {k.title(): v for k, v in expense_data.items()}

        overspending_alerts = []
        if current_user.income:
            wants_limit = current_user.income * 0.3
            wants_spent = (
                expense_data.get("entertainment", 0) + expense_data.get("shopping", 0)
            )
            if wants_spent > wants_limit:
                overspending_alerts.append(
                    f"Overspending in 'Wants': ₹{wants_spent} spent vs ₹{wants_limit} budget"
                )

        return jsonify({
            "total_spent": total_spent,
            "by_category": by_category,
            "alerts": overspending_alerts,
        })
    except Exception as e:
        logger.error("Summary fetch failed: %s", e)
        return jsonify({"error": f"Error fetching summary: {str(e)}"}), 500
