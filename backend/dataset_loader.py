import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_mutual_funds_df = None
_stocks_df = None


def load_mutual_funds():
    """Load mutual fund CSV (small file, safe at startup)."""
    global _mutual_funds_df
    if _mutual_funds_df is None:
        logger.info("Loading mutual fund dataset...")
        mf_path = os.path.join(BASE_DIR, "data", "Mutual_Funds.csv")
        if os.path.exists(mf_path):
            _mutual_funds_df = pd.read_csv(mf_path)
            logger.info("Mutual funds data loaded.")
        else:
            logger.warning("Mutual funds data not found.")
            _mutual_funds_df = pd.DataFrame()
    return _mutual_funds_df


def _load_stocks_dataframe():
    """Lazy-load all stock CSVs on first stock API request."""
    global _stocks_df
    if _stocks_df is not None:
        return _stocks_df

    logger.info("Loading historical stock datasets (first request)...")
    stocks_dir = os.path.join(BASE_DIR, "data", "stocks")
    if not os.path.exists(stocks_dir):
        logger.warning("Stocks directory not found.")
        _stocks_df = pd.DataFrame()
        return _stocks_df

    all_stock_files = [
        os.path.join(stocks_dir, f)
        for f in os.listdir(stocks_dir)
        if f.endswith('.csv')
    ]
    valid_files = [f for f in all_stock_files if os.path.getsize(f) > 0]

    if valid_files:
        _stocks_df = pd.concat([pd.read_csv(f) for f in valid_files], ignore_index=True)
        logger.info("Loaded %s historical stock files.", len(valid_files))
    else:
        logger.warning("No valid stock data files found.")
        _stocks_df = pd.DataFrame()

    return _stocks_df


def get_stock_data(symbol):
    """Return the last 100 rows for a given stock symbol."""
    df = _load_stocks_dataframe()
    if df is None or df.empty or "Symbol" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    symbol = symbol.upper()
    return df[df["Symbol"] == symbol].tail(100)


def get_stock_symbols():
    """Return sorted unique stock symbols."""
    df = _load_stocks_dataframe()
    if df is None or df.empty or "Symbol" not in df.columns:
        return []
    return sorted(df["Symbol"].astype(str).str.strip().str.upper().unique().tolist())


def load_all_datasets():
    """Backward-compatible loader used during app startup."""
    return {
        "mutual_funds": load_mutual_funds(),
        "nifty_50_historical": _stocks_df if _stocks_df is not None else pd.DataFrame(),
    }
