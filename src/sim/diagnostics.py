# Dynamic Conditional Correlation (DCC-GARCH) for simulation.
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data"))
sys.path.append(_DATA_DIR)
from loaders import load_daily  # noqa: E402

DEFAULT_INDUSTRIES = os.path.join(_DATA_DIR, "raw", "49_Industry_Portfolios_Daily.csv")
DEFAULT_FACTORS = os.path.join(_DATA_DIR, "raw", "F-F_Research_Data_5_Factors_2x3_daily.csv")


def build_returns(industries_path, factors_path, start=None, end=None):
    """Clean, balanced panel of daily industry returns (percent): aligned to the
    factor calendar, restricted to the period where every industry has data, with
    residual NaN rows dropped."""
    ind = load_daily(industries_path)
    fac = load_daily(factors_path)

    # align to the factor file's trading calendar
    common = ind.index.intersection(fac.index)
    ind = ind.loc[common]

    if start is not None:
        ind = ind.loc[ind.index >= pd.Timestamp(start)]
    if end is not None:
        ind = ind.loc[ind.index <= pd.Timestamp(end)]

    # balance: start at the latest first-valid date across columns, drop residual NaN rows
    first_valid = ind.apply(lambda s: s.first_valid_index()).max()
    ind = ind.loc[ind.index >= first_valid]
    n_before = len(ind)
    ind = ind.dropna(axis=0, how="any")
    if len(ind) < n_before:
        print(f"  dropped {n_before - len(ind)} rows with residual NaNs after balancing")
    return ind


def stationarity_report(returns, alpha=0.05, lb_lags=10, arch_lags=12):
    """Per-series DCC-GARCH suitability diagnostics: ADF + KPSS (stationarity),
    Ljung-Box (mean autocorr), Engle ARCH-LM (vol clustering), Jarque-Bera
    (normality). Returns a DataFrame with an ``ok`` flag (stationary + has ARCH)."""
    rows = {}
    for col in returns.columns:
        raw = returns[col].dropna()
        s = raw - raw.mean()  # demean for the variance/ARCH diagnostics

        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(s, autolag="AIC")

        # KPSS p-values are clipped to [0.01, 0.10]; silence the boundary warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")

        lb = acorr_ljungbox(s, lags=[lb_lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_p = float(lb["lb_pvalue"].iloc[0])

        arch_lm, arch_lm_p, *_ = het_arch(s, nlags=arch_lags)

        jb_stat, jb_p = stats.jarque_bera(s)

        stationary = (adf_p < alpha) and (kpss_p > alpha)
        has_arch = arch_lm_p < alpha
        rows[col] = {
            "n": int(raw.shape[0]),
            "mean": float(raw.mean()),
            "std": float(raw.std()),
            "min": float(raw.min()),
            "max": float(raw.max()),
            "skew": float(stats.skew(raw)),
            "exkurt": float(stats.kurtosis(raw)),  # excess kurtosis (Fisher)
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "adf_5pct": adf_crit["5%"],
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "kpss_5pct": kpss_crit["5%"],
            "lb_stat": lb_stat,
            "lb_p": lb_p,
            "arch_stat": arch_lm,
            "arch_p": arch_lm_p,
            "jb_stat": jb_stat,
            "jb_p": jb_p,
            "stationary": stationary,
            "has_arch": has_arch,
            "ok": stationary and has_arch,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def summarize_report(report, alpha=0.05):
    """Print a compact summary and flag any series unfit for DCC-GARCH."""
    n = len(report)
    n_stat = int(report["stationary"].sum())
    n_arch = int(report["has_arch"].sum())
    n_ok = int(report["ok"].sum())

    print(f"\nDiagnostics over {n} series (alpha = {alpha}):")
    print(f"  stationary (ADF reject & KPSS fail-to-reject) : {n_stat}/{n}")
    print(f"  ARCH effects present (Engle LM reject)        : {n_arch}/{n}")
    print(f"  fully OK for DCC-GARCH                         : {n_ok}/{n}")

    not_stationary = report.index[~report["stationary"]].tolist()
    no_arch = report.index[~report["has_arch"]].tolist()
    if not_stationary:
        print(f"\n  NOT stationary -> inspect before modelling: {not_stationary}")
    if no_arch:
        print(f"  No ARCH effects -> GARCH may be unnecessary: {no_arch}")
    if not not_stationary and not no_arch:
        print("\n  All series stationary with ARCH effects: safe to calibrate DCC-GARCH.")


def _print_table(df, cols, fmt):
    """Print a column subset with a per-call float format, full rows shown."""
    with pd.option_context("display.float_format", fmt,
                           "display.max_rows", None,
                           "display.width", 200):
        print(df[cols].to_string())


def main(industries_path, factors_path, start, end, out=None):
    print("Loading daily industry returns...")
    returns = build_returns(industries_path, factors_path, start=start, end=end)
    print(f"  panel: {returns.shape}  "
          f"{returns.index[0].date()} — {returns.index[-1].date()}")

    print("\nRunning stationarity / GARCH-suitability diagnostics...")
    report = stationarity_report(returns)

    # Descriptive moments (decimal formatting).
    print("\n--- descriptive moments (raw % returns) ---")
    _print_table(report, ["n", "mean", "std", "min", "max", "skew", "exkurt"],
                 lambda v: f"{v:.3f}")

    # Test statistics with their 5% critical values, where applicable.
    print("\n--- unit-root / stationarity (ADF H0: unit root; KPSS H0: stationary) ---")
    _print_table(report,
                 ["adf_stat", "adf_p", "adf_5pct",
                  "kpss_stat", "kpss_p", "kpss_5pct", "stationary"],
                 lambda v: f"{v:.4g}")

    # p-values in scientific notation so tiny ones don't collapse to 0.0000.
    print("\n--- autocorrelation / ARCH / normality (statistics and p-values) ---")
    _print_table(report,
                 ["lb_stat", "lb_p", "arch_stat", "arch_p",
                  "jb_stat", "jb_p", "has_arch", "ok"],
                 lambda v: f"{v:.3e}")

    summarize_report(report)

    if out:
        report.to_csv(out)
        print(f"\nfull report written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stationarity checks for DCC-GARCH calibration")
    ap.add_argument("--industries", default=DEFAULT_INDUSTRIES)
    ap.add_argument("--factors", default=DEFAULT_FACTORS)
    ap.add_argument("--start", default="1970-01-01",
                    help="sample start (YYYY-MM-DD); early FF data has many gaps")
    ap.add_argument("--end", default=None, help="sample end (YYYY-MM-DD)")
    ap.add_argument("--out", default=None,
                    help="optional path to write the full report as CSV")
    args = ap.parse_args()
    main(args.industries, args.factors, args.start, args.end, out=args.out)
