import logging

from flask import Blueprint, request, jsonify

from auth import register_user, login_user, token_required

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    required_fields = ['email', 'password', 'name']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"{field} is required"}), 400

    income = data.get('income')
    goal = data.get('goal')
    risk_profile = data.get('risk_profile')

    result, status_code = register_user(
        email=data['email'],
        password=data['password'],
        name=data['name'],
        income=income,
        goal=goal,
        risk_profile=risk_profile
    )

    return jsonify(result), status_code


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    if not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400

    result, status_code = login_user(
        email=data['email'],
        password=data['password']
    )

    return jsonify(result), status_code


@auth_bp.route("/me", methods=["GET"])
@token_required
def get_user_profile(current_user):
    return jsonify({
        'user': {
            'id': current_user.id,
            'email': current_user.email,
            'name': current_user.name,
            'income': current_user.income,
            'goal': current_user.goal,
            'risk_profile': current_user.risk_profile,
            'created_at': current_user.created_at.isoformat() if current_user.created_at else None
        }
    })


@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout(current_user):
    return jsonify({"message": "Logged out successfully"})
