"""
Configuration settings for Financial Goal Planner
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))


class Config:
    """Application configuration loaded from environment variables."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-change-me')

    _jwt_secret = os.environ.get('JWT_SECRET_KEY')
    if not _jwt_secret or _jwt_secret == 'your-secret-key-change-in-production':
        raise RuntimeError("JWT_SECRET_KEY not set. Add it to your .env file.")
    JWT_SECRET_KEY = _jwt_secret

    raw_url = os.environ.get('DATABASE_URL', 'sqlite:///finance.db')
    # Fix Supabase "postgres://" → "postgresql://"
    if raw_url.startswith('postgres://'):
        raw_url = raw_url.replace('postgres://', 'postgresql://', 1)
    DATABASE_URL = raw_url

    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Only apply pool settings for PostgreSQL, not SQLite
    IS_POSTGRES = DATABASE_URL.startswith('postgresql://')

    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.environ.get('SUPABASE_DB_POOL_SIZE', 5)),
        'max_overflow': int(os.environ.get('SUPABASE_DB_MAX_OVERFLOW', 10)),
        'pool_pre_ping': True,     # verify connection before use
        'pool_recycle': 300,       # recycle connections every 5 minutes
    } if IS_POSTGRES else {}

    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    ALLOW_SEED = os.environ.get('ALLOW_SEED', 'false').lower() == 'true'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # CORS configuration
    CORS_ORIGINS = [
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'http://localhost:5500',
        'http://127.0.0.1:5500',
        'file://',
        'null'
    ]


def get_config():
    """Get configuration class."""
    return Config
