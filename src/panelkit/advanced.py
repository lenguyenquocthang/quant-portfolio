"""
panel_adv.py — Robust inference & dynamic models for the NIM panel (N≈27 banks, T≈11 years).
Dependencies: numpy, pandas, scipy, statsmodels, linearmodels; (optional) pydynpd for dynamic GMM.

  1. cluster_se_ols / dk_se_ols : SEs clustered by bank, by year, and Driscoll–Kraay (pure numpy)
  2. size_distortion_mc          : Monte Carlo — false rejection rate (H0 true) for a year-only variable
  3. wild_cluster_p              : wild cluster bootstrap (Webb 6-point) when the number of clusters is small (T=11)
  4. ho_saunders_two_step        : step 1 (bank characteristics + year dummies) → step 2 (macro variables on T points)
  5. nickell_bias_mc             : Nickell bias of FE with lagged NIM; Anderson–Hsiao (IV) as a benchmark
  6. dynamic_gmm                 : Arellano–Bond/Blundell–Bond via pydynpd (with a shim for NumPy ≥ 2)
  7. leave_one_bank_out / winsorize : sensitivity to influential observations
  8. rolling_origin_eval         : out-of-sample evaluation by year (vs. the naive forecast NIM_{t-1}) (extreme NPL)
"""
from __future__ import annotations

import builtins
import math
import types
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS


# --------------------------------------------------------------------------- #
# 1. Robust SEs (pure numpy for fast Monte Carlo)
# --------------------------------------------------------------------------- #
def _ols(y: np.ndarray, X: np.ndarray):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    return b, y - X @ b, XtXi


def cluster_se_ols(y, X, groups) -> tuple[np.ndarray, np.ndarray]:
    """CR1: V = c (X'X)^-1 [Σ_g X_g' u_g u_g' X_g] (X'X)^-1, c = G/(G-1) · (n-1)/(n-k)."""
    b, u, XtXi = _ols(y, X)
    g = pd.factorize(groups)[0]
    G, n, k = g.max() + 1, len(y), X.shape[1]
    S = np.zeros((k, k))
    for j in range(G):
        s = X[g == j].T @ u[g == j]
        S += np.outer(s, s)
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(V))


def dk_se_ols(y, X, time, lags: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Driscoll–Kraay (1998): HAC (Bartlett) on the time series of moments h_t = Σ_i x_it u_it.
    Asymptotics are in large T — with T≈11 it is still biased downward; treat it as a lower bound on uncertainty."""
    b, u, XtXi = _ols(y, X)
    t = pd.factorize(time, sort=True)[0]
    T = t.max() + 1
    H = np.zeros((T, X.shape[1]))
    for s in range(T):
        H[s] = X[t == s].T @ u[t == s]
    m = lags if lags is not None else int(math.floor(4 * (T / 100) ** (2 / 9))) or 1
    S = H.T @ H
    for l in range(1, m + 1):
        w = 1 - l / (m + 1)
        Gm = H[l:].T @ H[:-l]
        S += w * (Gm + Gm.T)
    V = XtXi @ S @ XtXi * T / (T - 1)
    return b, np.sqrt(np.diag(V))


# --------------------------------------------------------------------------- #
# 2. Monte Carlo: why bank-clustered SEs "inflate" the significance of GDP/INF/COVID
# --------------------------------------------------------------------------- #
def size_distortion_mc(R: int = 400, N: int = 27, T: int = 11, shock_sd: float = 0.15, e_sd: float = 0.30,
                       seed: int = 7) -> pd.DataFrame:
    """H0 TRUE (coefficient on the macro variable g_t = 0). Measures the empirical rejection rate at the nominal 5% level.
    DGP: y_it = a_i + u_t + e_it, g_t ~ N(0,1) independent — only one g value per year."""
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(N), T); yr = np.tile(np.arange(T), N)
    rej = {"bank-cluster (z)": 0, "bank-cluster (t, N-1)": 0, "year-cluster (t, T-1)": 0,
           "Driscoll–Kraay (t, T-1)": 0, "iid OLS": 0, "year-means OLS (t, T-2)": 0}
    for _ in range(R):
        a = rng.normal(0, 0.6, N); u = rng.normal(0, shock_sd, T); g = rng.normal(0, 1, T)
        y = a[ent] + u[yr] + rng.normal(0, e_sd, N * T)
        X = np.column_stack([np.ones(N * T), g[yr]])
        b, se = cluster_se_ols(y, X, ent)
        tstat = b[1] / se[1]
        rej["bank-cluster (z)"] += abs(tstat) > 1.96
        rej["bank-cluster (t, N-1)"] += abs(tstat) > stats.t.ppf(0.975, N - 1)
        b2, se2 = cluster_se_ols(y, X, yr)
        rej["year-cluster (t, T-1)"] += abs(b2[1] / se2[1]) > stats.t.ppf(0.975, T - 1)
        b3, se3 = dk_se_ols(y, X, yr)
        rej["Driscoll–Kraay (t, T-1)"] += abs(b3[1] / se3[1]) > stats.t.ppf(0.975, T - 1)
        bb, uu, XtXi = _ols(y, X)
        s2 = uu @ uu / (N * T - 2)
        rej["iid OLS"] += abs(bb[1] / math.sqrt(s2 * XtXi[1, 1])) > 1.96
        ym = y.reshape(N, T).mean(0)                    # year means → T observations
        Xm = np.column_stack([np.ones(T), g])
        bm, um, Xi = _ols(ym, Xm)
        rej["year-means OLS (t, T-2)"] += abs(bm[1] / math.sqrt(um @ um / (T - 2) * Xi[1, 1])) > stats.t.ppf(0.975, T - 2)
    return pd.Series({k: v / R for k, v in rej.items()}, name="rejection rate (nominal 0.05)").to_frame()


# --------------------------------------------------------------------------- #
# 3. Wild cluster bootstrap-t (few clusters)
# --------------------------------------------------------------------------- #
_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_p(y, X, groups, j: int, B: int = 999, seed: int = 0) -> dict:
    """Wild cluster bootstrap-t with the H0 restriction b_j = 0 (Cameron–Gelbach–Miller 2008; Webb 6-point weights
    for G < 12). Returns the two-sided p-value. groups = year (T clusters) when testing macro variables."""
    rng = np.random.default_rng(seed)
    g = pd.factorize(groups)[0]; G = g.max() + 1
    b, se = cluster_se_ols(y, X, g)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br, ur, _ = _ols(y, Xr)
    fit_r = Xr @ br
    cnt = 0
    for _ in range(B):
        w = rng.choice(_WEBB, size=G)[g]
        yb = fit_r + ur * w
        bb, seb = cluster_se_ols(yb, X, g)
        cnt += abs(bb[j] / seb[j]) >= abs(t0)
    return {"t": float(t0), "p_wild": float((cnt + 1) / (B + 1)), "G": int(G)}


# --------------------------------------------------------------------------- #
# 4. Ho–Saunders (1981) two-step
# --------------------------------------------------------------------------- #
def ho_saunders_two_step(df: pd.DataFrame, y: str, bank_vars: list[str], macro: pd.DataFrame,
                         bank="bank", year="year") -> dict:
    """Step 1: NIM_it = Σ_t α_t D_t + γ'X_it + ε_it  (year-dummy coefficients α_t = 'pure spread').
    Step 2: α̂_t = δ0 + δ'M_t + η_t with M_t = macro variables (interest-rate σ, inflation, GDP, market structure) — only T observations;
    use WLS with weights 1/Var(α̂_t) (Hanushek) because α̂_t is a dependent variable measured with estimation error."""
    d = df.copy()
    D = pd.get_dummies(d[year], prefix="y", dtype=float)
    X1 = pd.concat([D, d[bank_vars]], axis=1)
    X1 = X1.to_numpy(); Y1 = d[y].to_numpy()
    b, se = cluster_se_ols(Y1, X1, d[bank].to_numpy())
    k = D.shape[1]
    alpha = pd.Series(b[:k], index=[int(c[2:]) for c in D.columns]); alpha_se = pd.Series(se[:k], index=alpha.index)
    M = macro.loc[alpha.index]
    X2 = np.column_stack([np.ones(len(M)), M.to_numpy()])
    w = 1.0 / alpha_se.to_numpy() ** 2
    Wh = np.sqrt(w)
    b2, u2, Xi = _ols(alpha.to_numpy() * Wh, X2 * Wh[:, None])
    s2 = u2 @ u2 / (len(M) - X2.shape[1])
    se2 = np.sqrt(np.diag(s2 * Xi))
    step2 = pd.DataFrame({"coef": b2, "se": se2, "p": 2 * stats.t.sf(np.abs(b2 / se2), len(M) - X2.shape[1])},
                         index=["const"] + list(M.columns))
    return {"pure_spread": alpha, "gamma": pd.Series(b[k:], index=bank_vars), "step2": step2}


# --------------------------------------------------------------------------- #
# 5. Nickell bias & Anderson–Hsiao
# --------------------------------------------------------------------------- #
def _dyn_panel(N, T, rho, beta, seed, sd_a=1.0, sd_e=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        a = rng.normal(0, sd_a); x_dev = rng.normal(0, 1); y = a / (1 - rho) + rng.normal(0, sd_e)
        for t in range(T + 3):
            x = rng.normal(8, 2)
            y = rho * y + beta * x + a + rng.normal(0, sd_e)
            rows.append((i, t, y, x))
    d = pd.DataFrame(rows, columns=["bank", "t", "y", "x"])
    return d[d.t >= 3].reset_index(drop=True)          # drop the burn-in period


def nickell_bias_mc(R: int = 200, N: int = 27, T: int = 11, rho: float = 0.5, beta: float = 0.1,
                    seed: int = 11) -> pd.Series:
    """Compare estimates of rho: Pooled OLS (upward bias), FE (downward bias ≈ -(1+rho)/(T-1)), Anderson–Hsiao IV."""
    out = {"pooled": [], "fe": [], "anderson_hsiao": []}
    for r in range(R):
        d = _dyn_panel(N, T, rho, beta, seed + r)
        d["ly"] = d.groupby("bank").y.shift(1); d["l2y"] = d.groupby("bank").y.shift(2)
        d = d.dropna(subset=["ly"])
        Xp = np.column_stack([np.ones(len(d)), d.ly, d.x]); out["pooled"].append(_ols(d.y.to_numpy(), Xp)[0][1])
        P = d.set_index(["bank", "t"])
        fe = PanelOLS(P.y, P[["ly", "x"]], entity_effects=True).fit()
        out["fe"].append(fe.params["ly"])
        dd = d.copy()
        for c in ("y", "ly", "x"):
            dd["d" + c] = dd.groupby("bank")[c].diff()
        dd["l2y"] = d.groupby("bank").y.shift(2)
        dd = dd.dropna(subset=["dy", "dly", "dx", "l2y"])
        iv = IV2SLS(dd.dy, dd[["dx"]], dd[["dly"]], dd[["l2y"]]).fit(cov_type="clustered", clusters=dd.bank)
        out["anderson_hsiao"].append(iv.params["dly"])
    res = pd.DataFrame(out)
    s = pd.concat([res.mean().rename("mean"), res.median().rename("median"), res.std().rename("std. dev.")], axis=1)
    s.loc["Nickell approx. FE", "mean"] = rho - (1 + rho) / (T - 1)
    s.attrs["true_rho"] = rho
    return s


# --------------------------------------------------------------------------- #
# 6. Dynamic GMM (pydynpd) — shim for NumPy ≥ 2.x
# --------------------------------------------------------------------------- #
def dynamic_gmm(df: pd.DataFrame, command: str, ids=("bank", "year")):
    """Example command (small N ⇒ collapsing is MANDATORY, keep #instruments < N):
         'nim L1.nim cap ldr npl | gmm(nim, 2:4) iv(cap ldr npl) | collapse nolevel timedumm'
       - nolevel = Arellano–Bond (difference GMM); drop nolevel = system GMM (requires Δx uncorrelated with α_i).
       - Report: Hansen J (p should be 0.1–0.9; p≈1 signals too many instruments), AR(1) significant, AR(2) NOT significant.
       - Windmeijer-corrected SEs are already included in the two-step results."""
    np.in1d = getattr(np, "in1d", np.isin)
    import pydynpd.specification_tests as st
    st.float = lambda x: builtins.float(np.asarray(x).reshape(-1)[0])
    st.math = types.SimpleNamespace(sqrt=lambda x: math.sqrt(builtins.float(np.asarray(x).reshape(-1)[0])))
    from pydynpd import regression
    return regression.abond(command, df, list(ids))


# --------------------------------------------------------------------------- #
# 7. Influential observations
# --------------------------------------------------------------------------- #
def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def leave_one_bank_out(panel: pd.DataFrame, y: str, xs: list[str], target: str) -> pd.DataFrame:
    """Re-estimate FE dropping each bank in turn: shows whether the `target` coefficient depends on a single bank (e.g. NPL=29.8%)."""
    rows = []
    for b in panel.index.get_level_values(0).unique():
        sub = panel.drop(index=b, level=0)
        r = PanelOLS(sub[y], sub[xs], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
        rows.append((b, r.params[target], r.pvalues[target]))
    t = pd.DataFrame(rows, columns=["bank_dropped", "coef", "p"]).set_index("bank_dropped")
    return t.sort_values("coef")


# --------------------------------------------------------------------------- #
# 8. Out-of-sample evaluation: rolling-origin by year
# --------------------------------------------------------------------------- #
def rolling_origin_eval(panel: pd.DataFrame, y: str, xs: list[str], first_test_year: int) -> pd.DataFrame:
    """For each test year t: estimate on years < t, forecast NIM_t CONDITIONAL on the realized X_t
    (this assesses out-of-sample explanatory power, not true forecasting). Compares:
      naive   : NIM_{i,t-1}  (pure persistence — the minimum benchmark the model must beat)
      pooled  : pooled OLS
      fe      : FE, with bank effects estimated from the training sample  alpha_i = ybar_i - xbar_i' b
    Returns RMSE/MAE by year; assessing differences with Diebold–Mariano across years is only descriptive (few years)."""
    d = panel.reset_index().rename(columns={panel.index.names[0]: "bank", panel.index.names[1]: "year"})
    years = sorted(d.year.unique())
    rows = []
    for t in [yr for yr in years if yr >= first_test_year]:
        tr, te = d[d.year < t], d[d.year == t]
        te = te[te.bank.isin(tr.bank.unique())]
        if te.empty or tr.year.nunique() < 3:
            continue
        # variables constant in the training sample (e.g. COVID before 2020) are not estimable → lstsq minimum-norm solution (coefficient 0)
        Xtr = np.column_stack([np.ones(len(tr)), tr[xs]]); ytr = tr[y].to_numpy()
        b_pool = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        Xte = np.column_stack([np.ones(len(te)), te[xs]])
        pred_pool = Xte @ b_pool
        # FE (within) — use only variables with within-bank variation
        dm = tr[[y] + xs].groupby(tr.bank).transform(lambda s: s - s.mean())
        keep = [x for x in xs if dm[x].std() > 1e-12]
        b_fe = np.linalg.lstsq(dm[keep].to_numpy(), dm[y].to_numpy(), rcond=None)[0]
        mu = tr.groupby("bank")[[y] + keep].mean()
        alpha = mu[y] - mu[keep].to_numpy() @ b_fe
        pred_fe = alpha.reindex(te.bank).to_numpy() + te[keep].to_numpy() @ b_fe
        last = tr[tr.year == tr.year.max()].set_index("bank")[y]
        pred_naive = last.reindex(te.bank).to_numpy()
        ok = ~np.isnan(pred_naive)
        yt = te[y].to_numpy()
        for name, p in (("naive", pred_naive), ("pooled", pred_pool), ("fe", pred_fe)):
            e = (yt - p)[ok]
            rows.append((t, name, float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))))
    out = pd.DataFrame(rows, columns=["year", "model", "RMSE", "MAE"])
    return out.pivot(index="year", columns="model", values="RMSE")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    warnings.simplefilter("ignore")
    print(size_distortion_mc(R=300))
    print(nickell_bias_mc(R=60))
