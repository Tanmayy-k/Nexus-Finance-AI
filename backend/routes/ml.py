import logging
from datetime import date

from flask import Blueprint, request, jsonify

from auth import token_required
from models import Expense, PredictionLog, db
from services.ml_service import (
    predict_score,
    get_cluster_users_payload,
    build_feature_df,
    get_model,
)
from services.shap_service import explain_score, split_factors

logger = logging.getLogger(__name__)

ml_bp = Blueprint('ml', __name__)


def _resolve_prediction_inputs(current_user, data=None):
    """Build feature values used for prediction and SHAP explanation."""
    data = data or {}

    monthly_income = data.get("monthly_income", current_user.income or 50000)

    first_day = date.today().replace(day=1)
    month_expenses = Expense.query.filter(
        Expense.user_id == current_user.id,
        Expense.created_at >= first_day,
    ).all()
    monthly_expense_total = sum(e.amount for e in month_expenses)

    if "investment_amount" in data:
        investment_amount = data.get("investment_amount")
    elif current_user.income:
        investment_amount = current_user.income * 0.1
    else:
        investment_amount = 0

    savings_rate = 0
    if monthly_income:
        savings_rate = max(0, (monthly_income - monthly_expense_total) / monthly_income)

    features_df = build_feature_df(
        monthly_income,
        monthly_expense_total,
        investment_amount,
        savings_rate=savings_rate,
    )
    features = features_df.iloc[0].to_dict()

    return features_df, features


def _compute_factors(features_df):
    """Run SHAP explanation; never raises — returns empty split on failure."""
    try:
        model = get_model()
        factors = explain_score(model, features_df)
        return split_factors(factors), factors
    except Exception as e:
        logger.warning("SHAP explanation skipped: %s", e)
        return {'positive': [], 'negative': []}, []


def _log_prediction(current_user, score, features_df, factors):
    try:
        log = PredictionLog(
            user_id=current_user.id,
            score=score,
            features=features_df.iloc[0].to_dict(),
            shap_values=factors,
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        logger.warning("Prediction log failed (non-critical): %s", e)
        db.session.rollback()


@ml_bp.route('/predict', methods=['POST'])
@token_required
def predict(current_user):
    data = request.json or {}

    try:
        features_df, features = _resolve_prediction_inputs(current_user, data)
        prediction = predict_score(features)

        factor_split, factors = _compute_factors(features_df)
        _log_prediction(current_user, prediction, features_df, factors)

        top_positive = factor_split['positive'][0]['label'] if factor_split['positive'] else ''
        top_negative = factor_split['negative'][0]['label'] if factor_split['negative'] else ''

        return jsonify({
            "prediction": prediction,
            "score": prediction,
            "factors": factor_split,
            "top_positive_label": top_positive,
            "top_negative_label": top_negative,
        })
    except RuntimeError as e:
        logger.error("Prediction failed — model not loaded: %s", e)
        return jsonify({"error": "ML model is not loaded."}), 500
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@ml_bp.route('/predict/explain', methods=['GET'])
@token_required
def predict_explain(current_user):
    try:
        features_df, _ = _resolve_prediction_inputs(current_user)
        factor_split, _ = _compute_factors(features_df)
        return jsonify({"factors": factor_split})
    except Exception as e:
        logger.error("Explanation failed: %s", e)
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 400


@ml_bp.route("/cluster_users", methods=["GET"])
@token_required
def cluster_users(current_user):
    try:
        result, status = get_cluster_users_payload(current_user)
        return jsonify(result), status
    except Exception as e:
        logger.error("Clustering failed: %s", e)
        return jsonify({"error": f"Error clustering users: {str(e)}"}), 500
