# init_db.py
from app import app, db

print("--- Running Database Init Script ---")

with app.app_context():
    db.create_all()
    print("Database tables created (if they didn't exist).")

print("--- Database Init Script Finished ---")
