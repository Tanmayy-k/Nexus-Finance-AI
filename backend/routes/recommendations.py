import logging
from datetime import date, timedelta

from flask import Blueprint, request, jsonify

from auth import token_required
from dataset_loader import load_mutual_funds, get_stock_data, get_stock_symbols
from models import Expense
from services.ml_service import recommend_for_user

logger = logging.getLogger(__name__)

recommendations_bp = Blueprint('recommendations', __name__)


@recommendations_bp.route("/recommendations", methods=["GET"])
@token_required
def recommendations(current_user):
    try:
        n = request.args.get("n", default=5, type=int)
        result = recommend_for_user(current_user.id, n=n)
        status = 200 if "error" not in result else 400
        return jsonify(result), status
    except Exception as e:
        logger.error("Recommendations failed: %s", e)
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500


@recommendations_bp.route("/investment", methods=["POST"])
@token_required
def investment(current_user):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    risk = data.get("risk", current_user.risk_profile or "medium").lower()
    savings = data.get("savings", current_user.income * 0.2 if current_user.income else 5000)

    suggestions = []
    if risk == "low":
        suggestions.append("Consider Fixed Deposits (FDs), Recurring Deposits (RDs), or Government Bonds.")
        if savings > 50000:
            suggestions.append("You have significant savings, consider PPF or low-risk Debt Mutual Funds.")
    elif risk == "high":
        suggestions.append("You can explore Equity Mutual Funds, direct Stocks, or ETFs.")
        if savings > 100000:
            suggestions.append(
                "Given your high savings, a diversified portfolio of large-cap stocks could be beneficial."
            )
    else:
        suggestions.append("A balanced approach: Consider SIPs in Index Funds or Balanced Mutual Funds.")
        if savings > 20000:
            suggestions.append("Start with an Index Fund SIP to build your wealth with moderate risk.")

    mf_data = load_mutual_funds()
    if mf_data is not None and not mf_data.empty:
        try:
            random_fund = mf_data['Scheme'].sample(n=1).iloc[0]
            suggestions.append(f"Based on our analysis, a good option is: {random_fund}.")
        except Exception:
            logger.warning("Could not sample random mutual fund for investment suggestion.")

    return jsonify({
        "risk_profile": risk,
        "savings": savings,
        "suggestions": suggestions,
    })


@recommendations_bp.route("/weekly_focus", methods=["GET"])
@token_required
def weekly_focus(current_user):
    try:
        cluster_label = current_user.cluster_id

        seven_days_ago = date.today() - timedelta(days=7)
        recent_expenses = Expense.query.filter(
            Expense.user_id == current_user.id,
            Expense.created_at >= seven_days_ago
        ).all()

        top_category = "General Savings"
        top_amount = 0
        if recent_expenses:
            category_spend = {}
            for e in recent_expenses:
                category_spend[e.category] = category_spend.get(e.category, 0) + e.amount

            if category_spend:
                top_category = max(category_spend, key=category_spend.get)
                top_amount = category_spend[top_category]

        if cluster_label == 0:
            focus_message = (
                f"Your focus is on building momentum! Your top expense was '{top_category}' "
                f"(₹{top_amount:,.0f}). Try to save an extra ₹500 this week!"
            )
        elif cluster_label == 1:
            focus_message = (
                f"Your biggest drain was '{top_category}' (₹{top_amount:,.0f}). "
                f"Let's try to cut spending in this one category by 10%!"
            )
        elif cluster_label == 2:
            focus_message = (
                f"You're in the 'High Spender' group. Your top category was '{top_category}' "
                f"(₹{top_amount:,.0f}). A great goal is to keep that specific category under ₹2000 this week."
            )
        else:
            focus_message = (
                f"Your top spending category last week was '{top_category}' "
                f"(₹{top_amount:,.0f}). See if you can reduce that by 10%!"
            )

        return jsonify({"focus_message": focus_message})

    except Exception as e:
        logger.error("Weekly focus failed: %s", e)
        return jsonify({"error": str(e)}), 500


@recommendations_bp.route("/api/stock_data/<symbol>", methods=["GET"])
def stock_data(symbol):
    try:
        stock_df = get_stock_data(symbol)
        if not stock_df.empty:
            return jsonify(stock_df.to_dict('records'))
        return jsonify({"error": f"No data found for {symbol}"}), 404
    except Exception as e:
        logger.error("Stock data fetch failed: %s", e)
        return jsonify({"error": f"Error fetching stock data: {str(e)}"}), 500


@recommendations_bp.route("/api/stocks/list", methods=["GET"])
def stock_list():
    try:
        symbols = get_stock_symbols()
        if symbols:
            return jsonify({"available_symbols": symbols})
        return jsonify({"error": "No stock symbols available"}), 404
    except Exception as e:
        logger.error("Stock list fetch failed: %s", e)
        return jsonify({"error": f"Error fetching stock list: {str(e)}"}), 500
