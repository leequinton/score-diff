import os

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from diagnostics import build_returns, DEFAULT_INDUSTRIES, DEFAULT_FACTORS

# Data loading
_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Length of the simulated covariance path. Decoupled from the in-sample length
# len(v): the DCC parameters are calibrated on history, but the simulated path can
# run arbitrarily long to give the diffusion model more distinct covariance states
# (pushing the overfitting threshold out). ~50k x 49 x 49 float64 ~ 1 GB on disk.
N_SIM_STEPS = 50_000
df = build_returns(DEFAULT_INDUSTRIES, DEFAULT_FACTORS, start="1969-07-01", end="2026-02-27")

# DCC-GARCH Fitting

# Part (1) univariate fitting to each series
# AR(1)-GARCH(1,1) with Student's t innovations
def fit_univariate(df, const_mean=("Guns",)):
    Z, results = {}, {}
    for col in df.columns:
        print(f"Fitting {col}...")
        mean = "Constant" if col in const_mean else "AR"
        lags = 0 if col in const_mean else 1
        am = arch_model(df[col], mean=mean, lags=lags, vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        Z[col] = res.std_resid
        results[col] = res
    return pd.DataFrame(Z).dropna(), results


# Part (2) DCC fitting to standardized resduals obtained
# Constant Conditional Correlation Estimate
def constant_corr(Z):
    v = Z.values
    Q_ = v.T @ v / len(v)
    d = np.sqrt(np.diag(Q_))
    R_ = Q_ / np.outer(d, d)
    return R_, Q_, v

# DCC
def dyn_corr(Qbar, v, alpha, beta):
    T, N = v.shape
    Q = Qbar.copy()
    R = np.empty((T, N, N))
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        R[t] = Q / np.outer(d, d)
        outer = np.outer(v[t], v[t])
        Q = Qbar + alpha * (outer - Qbar) + beta * (Q - Qbar)
    return R

# MLE DCC: Gaussian second-stage QML for (alpha, beta) on standardized residuals
def dcc_negloglik(params, Qbar, v):
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    T = v.shape[0]
    Q = Qbar.copy()
    ll = 0.0
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        try:
            c, low = cho_factor(R, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e10
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        quad = v[t] @ cho_solve((c, low), v[t], check_finite=False)
        ll += logdet + quad
        outer = np.outer(v[t], v[t])
        Q = Qbar + alpha * (outer - Qbar) + beta * (Q - Qbar)
    return 0.5 * ll / T  # mean NLL: keeps objective O(1) so SLSQP stays well-conditioned


def fit_dcc(Qbar, v, x0=(0.02, 0.97)):
    bounds = [(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)]
    cons = {"type": "ineq", "fun": lambda p: 1 - p[0] - p[1] - 1e-6}
    return minimize(dcc_negloglik, x0, args=(Qbar, v), method="SLSQP",
                    bounds=bounds, constraints=cons, options={"ftol": 1e-8})

# Simulating from DCC-GARCH
def _extract_params(results):
    cols = list(results.keys())
    N = len(cols)
    p = {k: np.zeros(N) for k in ("const", "ar1", "omega", "a", "b", "nu")}
    for i, col in enumerate(cols):
        par = results[col].params
        p["const"][i] = par.get("Const", 0.0)
        p["ar1"][i] = par.get(f"{col}[1]", 0.0)
        p["omega"][i] = par["omega"]
        p["a"][i] = par["alpha[1]"]
        p["b"][i] = par["beta[1]"]
        p["nu"][i] = par["nu"]
    return cols, p


def simulate_dcc(results, Qbar, alpha, beta, n_steps, seed=None, burn=500,
                 return_corr=False):
    rng = np.random.default_rng(seed)
    cols, p = _extract_params(results)
    N = len(cols)

    sigma2 = p["omega"] / (1 - p["a"] - p["b"])   # start at unconditional variance
    r_prev = p["const"] / (1 - p["ar1"])          # and unconditional AR(1) mean
    Q = Qbar.copy()

    ret = np.empty((n_steps, N))
    H = np.empty((n_steps, N, N))
    R_out = np.empty((n_steps, N, N)) if return_corr else None

    for t in range(burn + n_steps):
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        v = np.linalg.cholesky(R) @ (rng.standard_t(p["nu"]) * np.sqrt((p["nu"] - 2) / p["nu"]))

        sigma = np.sqrt(sigma2)
        eps = sigma * v
        r = p["const"] + p["ar1"] * r_prev + eps

        if t >= burn:
            k = t - burn
            ret[k] = r
            H[k] = (sigma[:, None] * R) * sigma[None, :]
            if return_corr:
                R_out[k] = R

        sigma2 = p["omega"] + p["a"] * eps**2 + p["b"] * sigma2
        r_prev = r
        Q = Qbar + alpha * (np.outer(v, v) - Qbar) + beta * (Q - Qbar)

    return (ret, H, R_out, cols) if return_corr else (ret, H, cols)


if __name__ == "__main__":
    Z, results = fit_univariate(df)
    print("Univariate fitting complete, standardized residuals:")
    print(Z.head())
    Rbar, Qbar, v = constant_corr(Z)
    print("Estimated constant correlation matrix:")
    print(Rbar)
    print("Calibrating DCC via Gaussian QMLE")
    opt = fit_dcc(Qbar, v)
    alpha, beta = opt.x
    print(f"alpha   = {alpha:.4f}")
    print(f"beta    = {beta:.4f}")
    print(f"loglik  = {-opt.fun * len(v):.1f}   converged={opt.success}")

    print(f"Simulating {N_SIM_STEPS} steps from the calibrated DCC-GARCH...")
    sim_ret, sim_H, cols = simulate_dcc(results, Qbar, alpha, beta,
                                        n_steps=N_SIM_STEPS, seed=0)
    print("sim returns:", sim_ret.shape, " sim cov:", sim_H.shape)
    eigmins = [np.linalg.eigvalsh(sim_H[k])[0] for k in (0, len(sim_H) // 2, -1)]
    print("min eigenvalue across sampled H_t:", float(min(eigmins)))
    np.save(os.path.join(_OUT_DIR, "sim_cov.npy"), sim_H)
    np.save(os.path.join(_OUT_DIR, "sim_returns.npy"), sim_ret)
    print("saved sim_cov.npy and sim_returns.npy")
