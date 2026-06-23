from flask import Flask
from flask_cors import CORS
from sqlalchemy import inspect, text

from config import Config
from logger_config import setup_logging
from models import db
from dataset_loader import load_mutual_funds
from routes.health import health_bp
from routes.auth import auth_bp
from routes.expenses import expenses_bp
from routes.budget import budget_bp
from routes.ml import ml_bp
from routes.recommendations import recommendations_bp
from routes.analytics import analytics_bp
from routes.copilot import copilot_bp
from services.ml_service import run_clustering_job, load_model


def _ensure_schema():
    """Apply lightweight SQLite patches until Alembic migrations are run."""
    inspector = inspect(db.engine)
    if 'user' in inspector.get_table_names():
        user_columns = {col['name'] for col in inspector.get_columns('user')}
        if 'cluster_id' not in user_columns:
            with db.engine.begin() as conn:
                conn.execute(text('ALTER TABLE user ADD COLUMN cluster_id INTEGER'))


def create_app():
    setup_logging()
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = Config.SQLALCHEMY_ENGINE_OPTIONS
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    CORS(
        app,
        origins=[
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "https://nexus-finance-ai-umber.vercel.app",
        ],
        supports_credentials=True,
    )

    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(expenses_bp)
    app.register_blueprint(budget_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(copilot_bp)

    with app.app_context():
        db.create_all()
        _ensure_schema()
        load_mutual_funds()
        load_model()
        run_clustering_job()

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=Config.FLASK_ENV == 'development')
