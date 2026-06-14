import logging
from datetime import datetime

from flask import Blueprint, jsonify

from dataset_loader import load_mutual_funds

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    })


@health_bp.route("/", methods=["GET"])
def home():
    mf_data = load_mutual_funds()

    if mf_data is not None and not mf_data.empty:
        dataset_preview = mf_data.head(3).to_dict()
    else:
        dataset_preview = {"message": "No datasets loaded."}

    return jsonify({
        "message": "Backend with DB + Dataset is working!",
        "dataset_preview": dataset_preview,
    })
