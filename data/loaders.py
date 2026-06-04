"""CSV loaders for the Ken French daily files.

Auto-detects the header line, keeps only the value-weighted block (dropping the
equal-weighted / annual section and footer), and maps the -99.99/-999 missing
codes to NaN. Used by the DCC-GARCH simulation in src/sim.
"""
import numpy as np
import pandas as pd


def _detect_skiprows(filepath):
    """Return skiprows so that the line before the first YYYYMMDD row is used as header."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            first = line.strip().split(",")[0].strip()
            if first.isdigit() and len(first) == 8:
                return i - 1
    raise ValueError(f"No YYYYMMDD date column found in {filepath}")


def load_daily(filepath):
    skiprows = _detect_skiprows(filepath)
    df = pd.read_csv(filepath, skiprows=skiprows, header=0, low_memory=False)
    df = df.rename(columns={df.columns[0]: "Date"})

    # drop equal-weighted / annual section that follows the value-weighted block
    non_date = ~df["Date"].astype(str).str.strip().str.match(r"^\d{8}$", na=False)
    if non_date.any():
        df = df.iloc[:non_date.idxmax()]

    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("Date").astype(float)
    df = df.replace(-99.99, np.nan).replace(-999.0, np.nan)
    return df
