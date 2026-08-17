"""Loading and preparing uploaded time-series signal data."""

import io

import numpy as np
import pandas as pd


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Read an uploaded .xlsx or .csv file into a DataFrame."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))


def find_time_column(df: pd.DataFrame) -> str:
    """Return the column that represents the X-axis (time)."""
    for col in df.columns:
        if "time" in str(col).lower():
            return col
    # Fall back to the first column if nothing is explicitly named "time".
    return df.columns[0]


def find_signal_columns(df: pd.DataFrame, time_col: str) -> list[str]:
    """Return Y columns, excluding the time column and any column that is all 1s."""
    signal_cols = []
    for col in df.columns:
        if col == time_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        if np.allclose(series.to_numpy(), 1.0):
            continue
        signal_cols.append(col)
    return signal_cols
