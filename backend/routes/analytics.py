import logging

from flask import Blueprint, jsonify

from auth import token_required
from services.anomaly_service import detect_anomalies
from services.forecasting_service import forecast_next_month
from services.insight_service import generate_insights

logger = logging.getLogger(__name__)

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/forecast', methods=['GET'])
@token_required
def forecast(current_user):
    result = forecast_next_month(current_user.id)
    return jsonify(result), 200


@analytics_bp.route('/anomalies', methods=['GET'])
@token_required
def anomalies(current_user):
    result = detect_anomalies(current_user.id)
    return jsonify(result), 200


@analytics_bp.route('/insights', methods=['GET'])
@token_required
def insights(current_user):
    result = generate_insights(current_user.id)
    return jsonify(result), 200
