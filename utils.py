# utils.py
from datetime import datetime
import pandas as pd
from typing import List, Tuple

DATE_COLS = ['po_date','etd','eta','atd','ata']

def parse_dates(df: pd.DataFrame):
    """
    Parse expected date columns safely (if present).
    """
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return None

def read_csv_safe(path: str, required_cols: List[str] = None, nrows: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read CSV and validate required columns.
    Returns (df, missing_columns_list).
    """
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, nrows=nrows)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}: {e}")

    missing = []
    if required_cols:
        present = set(df.columns.tolist())
        expected = set(required_cols)
        missing = list(expected - present)
    return df, missing
