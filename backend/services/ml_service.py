import logging
import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dataset_loader import load_mutual_funds
from models import User, db

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "finance_model.pkl")

_model = None
FEATURE_NAMES = []

FEATURE_COLUMNS = [
    "monthly_income",
    "monthly_expense_total",
    "investment_amount",
    "savings_rate",
]


def train_model():
    logger.info("Training model on dataset...")
    dataset_path = os.path.join(BASE_DIR, "data", "personal_finance_tracker_dataset.csv")
    df_tracker = pd.read_csv(dataset_path)
    if df_tracker.empty:
        logger.error("Dataset is empty. Skipping model training.")
        return None

    available_features = [c for c in FEATURE_COLUMNS if c in df_tracker.columns]
    if available_features != FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in available_features]
        logger.warning(
            "Dataset does not contain all expected feature columns. "
            "Using available features: %s; missing: %s",
            available_features,
            missing,
        )
    if not available_features:
        logger.error("No usable feature columns found in dataset.")
        return None

    X = df_tracker[available_features]
    y = df_tracker["financial_advice_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    global FEATURE_NAMES
    FEATURE_NAMES = list(X.columns)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    logger.info("Model trained — R²: %.3f, MAE: %.3f", r2, mae)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model training complete. Saved as finance_model.pkl")
    return model


def load_model():
    global _model, FEATURE_NAMES
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        loaded_model = joblib.load(MODEL_PATH)
        if not _is_model_feature_aligned(loaded_model):
            logger.info("Model feature mismatch detected. Retraining model to align with expected feature set.")
            _model = train_model()
        else:
            logger.info("Model loaded from finance_model.pkl")
            _model = loaded_model
            if hasattr(_model, 'feature_names_in_'):
                FEATURE_NAMES = list(_model.feature_names_in_)
    else:
        _model = train_model()
    return _model


def get_model():
    """Return the loaded model. Load if not already in memory."""
    return load_model()


def get_feature_names():
    """Return ordered feature names used during training."""
    if not FEATURE_NAMES:
        model = load_model()
        if model is not None and hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
    return FEATURE_NAMES


def build_feature_df(monthly_income, monthly_expense_total, investment_amount, savings_rate=None):
    """Build a single-row DataFrame with columns matching FEATURE_NAMES."""
    if savings_rate is None:
        savings_rate = (
            (monthly_income - monthly_expense_total) / monthly_income
            if monthly_income > 0 else 0.0
        )
    data = {
        'monthly_income': [monthly_income],
        'monthly_expense_total': [monthly_expense_total],
        'investment_amount': [investment_amount],
        'savings_rate': [savings_rate],
    }
    names = get_feature_names() or FEATURE_COLUMNS
    filtered = {k: v for k, v in data.items() if k in names}
    return pd.DataFrame(filtered)[names]


def predict_score(features: dict) -> float:
    model = load_model()
    if model is None:
        raise RuntimeError("ML model is not loaded.")

    feature_columns = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else FEATURE_COLUMNS
    row = {col: features.get(col, 0) for col in feature_columns}
    X_new = pd.DataFrame([row], columns=feature_columns)
    prediction = model.predict(X_new)[0]
    return round(float(prediction), 2)


def _is_model_feature_aligned(model) -> bool:
    if not hasattr(model, 'feature_names_in_'):
        logger.warning('Loaded model has no feature_names_in_; retraining required.')
        return False

    trained_features = list(model.feature_names_in_)
    if trained_features != FEATURE_COLUMNS:
        logger.warning(
            'Loaded model features do not match expected features. ' \
            'Trained: %s; expected: %s',
            trained_features,
            FEATURE_COLUMNS,
        )
        return False
    return True


def _user_features_df(users):
    return pd.DataFrame([{
        "id": u.id,
        "income": float(u.income or 0),
        "goal": 1 if (u.goal and str(u.goal).strip()) else 0,
        "risk": {"low": 0, "medium": 1, "high": 2}.get((u.risk_profile or "").lower(), 1),
    } for u in users])


def _get_user_cluster_and_peers(user_id, force_refit=False):
    users = User.query.all()
    if not users or len(users) < 3:
        return None, []

    all_have_clusters = all(u.cluster_id is not None for u in users)

    if not force_refit and all_have_clusters:
        user = User.query.get(user_id)
        if not user or user.cluster_id is None:
            return None, []
        cluster_label = int(user.cluster_id)
        peer_ids = [
            u.id for u in users
            if u.cluster_id == cluster_label and u.id != user_id
        ]
        return cluster_label, peer_ids

    df = _user_features_df(users)
    k = min(3, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df[["income", "risk", "goal"]].values)
    df["cluster"] = labels

    for user in users:
        row = df[df["id"] == user.id]
        if not row.empty:
            user.cluster_id = int(row["cluster"].iloc[0])
    db.session.commit()
    logger.info("KMeans clustering complete; cluster_id saved for %s users.", len(users))

    row = df[df["id"] == user_id]
    if row.empty:
        return None, []
    cluster_label = int(row["cluster"].iloc[0])
    peer_ids = df[df["cluster"] == cluster_label]["id"].tolist()
    peer_ids = [pid for pid in peer_ids if pid != user_id]
    return cluster_label, peer_ids


def run_clustering_job():
    """Run KMeans once and persist cluster_id for all users."""
    users = User.query.all()
    if not users or len(users) < 3:
        logger.info("Skipping clustering job — fewer than 3 users.")
        return

    if all(u.cluster_id is not None for u in users):
        logger.info("All users already have cluster_id; skipping refit.")
        return

    _get_user_cluster_and_peers(users[0].id, force_refit=True)


def _find_best_return_column(df):
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["1y", "1 yr", "1yr", "1-year", "return", "returns", "3y", "3yr", "aum"])
    ]
    numeric_candidates = []
    for c in candidates:
        try:
            pd.to_numeric(df[c].dropna().iloc[:5])
            numeric_candidates.append(c)
        except Exception:
            continue
    return numeric_candidates[0] if numeric_candidates else None


def _content_based_mf_recs(user, n=5):
    mf_data = load_mutual_funds()
    if mf_data is None or mf_data.empty:
        return []

    df = mf_data.copy()
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    risk_cols = [
        c for c in df.columns
        if c.lower() in ("risk", "risk_profile", "risk level", "risk_level")
    ]
    risk_col = risk_cols[0] if risk_cols else None
    return_col = _find_best_return_column(df)

    filtered = df
    user_risk = (user.risk_profile or "").lower()
    if risk_col and user_risk:
        try:
            filtered = df[df[risk_col].astype(str).str.lower().str.contains(user_risk.split()[0])]
        except Exception:
            filtered = df

    if return_col:
        filtered = filtered.copy()
        filtered[return_col] = pd.to_numeric(filtered[return_col], errors="coerce")
        filtered = filtered.sort_values(by=return_col, ascending=False)

    key_col = "Scheme" if "Scheme" in filtered.columns else (
        filtered.columns[0] if len(filtered.columns) > 0 else None
    )
    if key_col is None:
        return []

    recs = []
    for _, row in filtered.iterrows():
        name = str(row[key_col])
        reason = (
            "Matched by risk & top return" if return_col and risk_col
            else ("Top performer" if return_col else "Popular fund")
        )
        recs.append({"scheme": name, "reason": reason})
        if len(recs) >= n:
            break
    return recs


def _collaborative_recs(peer_ids, n=3):
    mf_data = load_mutual_funds()
    if mf_data is None or mf_data.empty:
        return []

    df = mf_data.copy()
    return_col = _find_best_return_column(df)
    if return_col:
        df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
        df = df.sort_values(by=return_col, ascending=False)
    key_col = "Scheme" if "Scheme" in df.columns else df.columns[0]
    recs = []
    for _, row in df.iterrows():
        recs.append({
            "scheme": str(row[key_col]),
            "reason": "Top dataset performer / peer proxy",
        })
        if len(recs) >= n:
            break
    return recs


def recommend_for_user(user_id, n=5):
    user = User.query.get(user_id)
    if not user:
        return {"error": "user not found"}

    cluster_label = user.cluster_id
    peer_ids = []
    if cluster_label is not None:
        peer_ids = [
            u.id for u in User.query.filter_by(cluster_id=cluster_label).all()
            if u.id != user_id
        ]
    else:
        cluster_label, peer_ids = _get_user_cluster_and_peers(user_id)

    c_n = max(1, int(round(n * 0.6)))
    coll_n = n - c_n

    content_recs = _content_based_mf_recs(user, c_n)
    collab_recs = _collaborative_recs(peer_ids, coll_n)

    combined = []
    seen = set()
    for r in (content_recs + collab_recs):
        if r["scheme"] not in seen:
            combined.append(r)
            seen.add(r["scheme"])
        if len(combined) >= n:
            break

    if len(combined) < n:
        if (user.risk_profile or "").lower() == "low":
            generic = ["Fixed Deposits (FD)", "PPF / Debt Mutual Funds", "Recurring Deposit (RD)"]
        elif (user.risk_profile or "").lower() == "high":
            generic = ["Index Fund SIP", "Large-cap Equity Mutual Funds", "ETFs / Stocks"]
        else:
            generic = ["Balanced Mutual Funds", "Index Fund SIP"]
        for g in generic:
            if len(combined) >= n:
                break
            if g not in seen:
                combined.append({
                    "scheme": g,
                    "reason": "Safe generic suggestion for your risk profile",
                })
                seen.add(g)

    return {
        "user_id": user_id,
        "cluster": int(cluster_label) if cluster_label is not None else None,
        "recommendations": combined,
    }


def get_cluster_users_payload(current_user):
    users = User.query.all()
    if not users or len(users) < 2:
        return {"error": "Not enough users to cluster."}, 400

    if all(u.cluster_id is not None for u in users):
        records = [{
            "id": u.id,
            "income": u.income or 0,
            "goal": 1 if u.goal else 0,
            "risk": {"low": 0, "medium": 1, "high": 2}.get((u.risk_profile or "").lower(), 1),
            "cluster": int(u.cluster_id),
        } for u in users]
    else:
        df = _user_features_df(users)
        kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(df[["income", "goal", "risk"]])
        for user in users:
            row = df[df["id"] == user.id]
            if not row.empty:
                user.cluster_id = int(row["cluster"].iloc[0])
        db.session.commit()
        records = df.to_dict(orient="records")

    current_user_cluster = next(
        (r["cluster"] for r in records if r["id"] == current_user.id),
        None,
    )
    if current_user_cluster is None:
        return {"error": "User not found in cluster data."}, 400

    return {
        "clusters": records,
        "current_user_cluster": int(current_user_cluster),
    }, 200
