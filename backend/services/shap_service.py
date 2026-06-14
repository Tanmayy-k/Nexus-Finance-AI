import logging

import pandas as pd
import shap

logger = logging.getLogger(__name__)

_explainer = None


def _get_explainer(model):
    """Build TreeExplainer once and cache it."""
    global _explainer
    if _explainer is None:
        logger.info("Building SHAP TreeExplainer...")
        _explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer ready.")
    return _explainer


def explain_score(model, features_df: pd.DataFrame) -> list[dict]:
    """
    Compute SHAP values for one prediction row.

    Returns a list of factor dicts sorted by absolute impact.
    Positive impact → helped the score.
    Negative impact → hurt the score.
    """
    try:
        explainer = _get_explainer(model)
        shap_values = explainer.shap_values(features_df)

        raw = shap_values[0] if hasattr(shap_values[0], '__len__') else shap_values

        label_map = {
            'monthly_income': 'Monthly Income',
            'monthly_expense_total': 'Total Monthly Expenses',
            'investment_amount': 'Investment Amount',
            'savings_rate': 'Savings Rate',
        }

        factors = []
        for feature, impact in zip(features_df.columns, raw):
            factors.append({
                'feature': feature,
                'label': label_map.get(feature, feature.replace('_', ' ').title()),
                'impact': round(float(impact), 2),
            })

        factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        return factors

    except Exception as e:
        logger.error("SHAP explanation failed: %s", e)
        return []


def split_factors(factors: list[dict]) -> dict:
    """Split factors into positive and negative contributors (top 3 each)."""
    positive = [f for f in factors if f['impact'] > 0][:3]
    negative = [f for f in factors if f['impact'] < 0][:3]
    return {'positive': positive, 'negative': negative}
