import logging

from flask import Blueprint, jsonify, request

from auth import token_required
from services.context_service import build_financial_context
from services.gemini_service import ask_copilot, is_gemini_configured

logger = logging.getLogger(__name__)

copilot_bp = Blueprint('copilot', __name__, url_prefix='/copilot')

MAX_MESSAGE_LENGTH = 500


@copilot_bp.route('/status', methods=['GET'])
@token_required
def status(current_user):
    mode = 'gemini' if is_gemini_configured() else 'fallback'
    return jsonify({'mode': mode}), 200


@copilot_bp.route('/chat', methods=['POST'])
@token_required
def chat(current_user):
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'message is required'}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            'error': f'Message too long. Max {MAX_MESSAGE_LENGTH} characters.',
        }), 400

    context = build_financial_context(current_user)
    result = ask_copilot(message, context)

    return jsonify({
        'reply': result['reply'],
        'mode': result['mode'],
    }), 200
