"""
panel_nim.py — Panel-data toolkit for bank NIM regressions (small N, short T).
Dependencies: numpy, pandas, scipy, statsmodels, linearmodels (pip install linearmodels).

Contents
  1. make_panel / audit_dummy / macro_identification : data checks & dummy-variable coding (catch errors before estimation)
  2. fit_models      : Pooled / FE / RE / Two-way FE, standard errors clustered by bank
  3. selection_tests : F (poolability), Breusch–Pagan LM (chibar2), classic Hausman, Mundlak (robust)
  4. diagnostics     : Wooldridge (serial correlation), Modified Wald (Greene), Pesaran CD, VIF
  5. simulate_bank_panel : DGP with known true parameters for testing the whole pipeline
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects


# --------------------------------------------------------------------------- #
# 1. Data standardization & validation
# --------------------------------------------------------------------------- #
def make_panel(df: pd.DataFrame, entity: str, time: str, cols: list[str],
               min_obs: int = 3) -> pd.DataFrame:
    """Return a sorted DataFrame with a MultiIndex (entity, time), after checking for common data errors."""
    d = df[[entity, time] + cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    if d.duplicated([entity, time]).any():
        raise ValueError("Duplicate (entity, time) keys: aggregate/drop duplicates before estimation.")
    n_before = len(d)
    d = d.dropna()
    if len(d) < n_before:
        warnings.warn(f"Dropped {n_before - len(d)} observations with missing data (listwise deletion).")
    cnt = d.groupby(entity)[time].transform("size")
    d = d[cnt >= min_obs]                      # entities with too few observations do not contribute to the within estimator
    d = d.set_index([entity, time]).sort_index()
    t_per = d.groupby(level=0).size()
    if t_per.nunique() > 1:
        warnings.warn(f"Unbalanced panel: T_i ranges from {t_per.min()} to {t_per.max()} "
                      "(check for survivorship bias: banks that merged/consolidated/listed late).")
    return d


def audit_dummy(panel: pd.DataFrame, col: str, expected_on: list, time_level: int = 1) -> pd.DataFrame:
    """Audit a dummy variable (e.g. COVID): observed value by year versus the expected value.
    Catches: {1,2} coding instead of {0,1}; wrong years assigned; reversed dummy."""
    s = panel[col].groupby(level=time_level).agg(["mean", "min", "max", "size"])
    vals = sorted(set(np.round(panel[col].unique(), 12)))
    s["expected_on"] = s.index.isin(expected_on)
    if vals != [0, 1]:
        warnings.warn(f"{col} takes values {vals}, not {{0,1}}: the coefficient is the effect of "
                      f"{vals[-1]} relative to {vals[0]} (and the intercept is shifted).")
    on = vals[-1]
    s["flagged_on"] = np.isclose(s["mean"], on)
    mismatch = s.index[s["expected_on"] != s["flagged_on"]].tolist()
    if mismatch:
        warnings.warn(f"{col}: coding differs from expectation in years {mismatch} → the coefficient may have the wrong sign/be meaningless.")
    return s


def macro_identification(panel: pd.DataFrame, macro_cols: list[str], time_level: int = 1) -> dict:
    """A variable that varies only over time carries as many independent pieces of information as periods T, not N*T."""
    g = panel[macro_cols].groupby(level=time_level)
    within_t_var = g.std().max().max()          # =0 if all banks share the same value within a year
    M = g.mean().to_numpy()
    M1 = np.column_stack([np.ones(len(M)), M])
    return {"n_periods": M.shape[0],
            "constant_within_period": bool(within_t_var < 1e-9),
            "rank_with_const": int(np.linalg.matrix_rank(M1)),
            "max_regressors_if_time_dummies": M.shape[0] - 1,
            "note": "If the three macro variables + constant have rank = k+1 ≤ T, the coefficients are identified "
                    "from only T data points; adding more than T-1 macro variables/year dummies causes perfect collinearity."}


# --------------------------------------------------------------------------- #
# 2. Estimation
# --------------------------------------------------------------------------- #
@dataclass
class Fits:
    pooled: object
    fe: object
    re: object
    twfe: object | None
    y: pd.Series
    X: pd.DataFrame


def fit_models(panel: pd.DataFrame, y: str, xs: list[str], time_effects: bool = True) -> Fits:
    """All models use entity-clustered SEs (robust to heteroskedasticity + within-bank serial correlation)."""
    Y = panel[y]
    X = sm.add_constant(panel[xs], has_constant="add")
    pooled = PooledOLS(Y, X).fit(cov_type="clustered", cluster_entity=True)
    fe = PanelOLS(Y, X, entity_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    re = RandomEffects(Y, X).fit(cov_type="clustered", cluster_entity=True)
    twfe = None
    if time_effects:
        # time-invariant-across-entities (time-only) variables are fully absorbed → drop_absorbed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the 'absorbed' warning is intentional
            twfe = PanelOLS(Y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
                cov_type="clustered", cluster_entity=True)
    return Fits(pooled, fe, re, twfe, Y, X)


def coef_table(fits: Fits) -> pd.DataFrame:
    out = {}
    for name in ("pooled", "fe", "re", "twfe"):
        r = getattr(fits, name)
        if r is None:
            continue
        out[name + "_b"] = r.params
        out[name + "_p"] = r.pvalues
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 3. Model selection tests
# --------------------------------------------------------------------------- #
def breusch_pagan_lm(pooled_resid: pd.Series) -> dict:
    """Breusch–Pagan LM test of H0: sigma_u^2 = 0 (no entity effects), formula for unbalanced panels.
    LM = (sum T_i)^2 / (2 sum T_i (T_i-1)) * [ sum_i (sum_t e_it)^2 / sum e^2 - 1 ]^2  ~ chi2(1) under H0.
    One-sided 'chibar2' p-value as in Stata xttest0: only when the bracket [..] is positive."""
    e = pooled_resid
    g = e.groupby(level=0)
    Ti = g.size().to_numpy(float)
    num = (g.sum() ** 2).sum()
    den = (e ** 2).sum()
    bracket = num / den - 1.0
    lm = (Ti.sum() ** 2) / (2.0 * (Ti * (Ti - 1)).sum()) * bracket ** 2
    p_chi2 = stats.chi2.sf(lm, 1)
    p_chibar = 0.5 * p_chi2 if bracket > 0 else 1.0
    return {"LM": float(lm), "p_chi2(1)": float(p_chi2), "p_chibar2": float(p_chibar)}


def hausman_classic(fits: Fits, vars_: list[str], sigmamore: bool = True) -> dict:
    """H = d'(V_FE - V_RE)^{-1} d. Valid ONLY if RE is efficient under H0 (homoskedasticity, no serial correlation).
    Uses non-clustered SEs (the condition of the original test). sigmamore=True (as in Stata): use the RE estimate of
    sigma_eps^2 for both covariance matrices → V_FE - V_RE is more often positive definite. If still not PSD → pinv + warning."""
    Y, X = fits.y, fits.X
    fe_u = PanelOLS(Y, X, entity_effects=True, drop_absorbed=True).fit()
    re_u = RandomEffects(Y, X).fit()
    scale = (float(re_u.variance_decomposition["Residual"]) / float(fe_u.s2)) if sigmamore else 1.0
    d = (fe_u.params[vars_] - re_u.params[vars_]).to_numpy()
    V = (scale * fe_u.cov.loc[vars_, vars_] - re_u.cov.loc[vars_, vars_]).to_numpy()
    eig = np.linalg.eigvalsh((V + V.T) / 2)
    pd_ok = bool(eig.min() > 1e-12)
    H = float(d @ np.linalg.pinv(V) @ d)
    if not pd_ok:
        warnings.warn("V_FE - V_RE is not positive definite → the Hausman statistic is unreliable (Stata's 'negative chi2' problem); "
                      "use the robust Mundlak test.")
    df_ = len(vars_)
    return {"H": H, "df": df_, "p": float(stats.chi2.sf(max(H, 0), df_)), "psd": pd_ok}


def mundlak_test(panel: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Mundlak (1978) / Wooldridge test: add the bank-level means of the bank-varying variables
    to the RE model; H0: coefficients on the means = 0 (≡ RE is consistent). Uses a robust (clustered) Wald test → valid under
    heteroskedasticity/serial correlation — in contrast to the classic Hausman test."""
    d = panel[[y] + xs].copy()
    # time-only variables: bank means are identical across banks → collinear with the constant
    cross_var = [x for x in xs if d.groupby(level=1)[x].std().max() > 1e-9 or d.groupby(level=0)[x].mean().std() > 1e-9]
    bar_vars = [x for x in cross_var if d.groupby(level=0)[x].mean().std() > 1e-9]
    bars = d.groupby(level=0)[bar_vars].transform("mean").add_suffix("_bar")
    X = sm.add_constant(pd.concat([d[xs], bars], axis=1), has_constant="add")
    res = RandomEffects(d[y], X).fit(cov_type="clustered", cluster_entity=True)
    b = res.params[bars.columns].to_numpy()
    V = res.cov.loc[bars.columns, bars.columns].to_numpy()
    W = float(b @ np.linalg.solve(V, b))
    k = len(b)
    # few clusters N → use F(k, N-1) instead of chi2 to be conservative
    N = d.index.get_level_values(0).nunique()
    p_chi2 = float(stats.chi2.sf(W, k))
    p_F = float(stats.f.sf(W / k, k, N - 1))
    return {"W": W, "df": k, "p_chi2": p_chi2, "p_F(k,N-1)": p_F, "vars": list(bars.columns)}


# --------------------------------------------------------------------------- #
# 4. Residual diagnostics
# --------------------------------------------------------------------------- #
def wooldridge_ar(panel: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Wooldridge (2002, §10.6.3): regress in first differences; if e_it is serially uncorrelated then
    corr(Δe_it, Δe_i,t-1) = -0.5. H0: rho = -0.5 (t with N-1 degrees of freedom, clustered SE)."""
    d = panel[[y] + xs].copy()
    # keep only variables that vary over time after differencing (drop constants)
    dd = d.groupby(level=0).diff().dropna()
    keep = [x for x in xs if dd[x].std() > 1e-12]
    r = sm.OLS(dd[y], dd[keep]).fit()
    e = pd.Series(r.resid, index=dd.index)
    lag = e.groupby(level=0).shift(1)
    m = pd.concat([e, lag], axis=1, keys=["e", "lag"]).dropna()
    g = m.index.get_level_values(0)
    fit = sm.OLS(m["e"], m[["lag"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    rho, se = float(fit.params["lag"]), float(fit.bse["lag"])
    N = pd.Series(g).nunique()
    t = (rho + 0.5) / se
    return {"rho_hat": rho, "se": se, "F": t ** 2, "p": float(2 * stats.t.sf(abs(t), N - 1))}


def modified_wald(fe_resid: pd.Series, k: int) -> dict:
    """Greene (2000): test for groupwise heteroskedasticity (differing error variances across entities).
    W = sum_i (s_i^2 - s^2)^2 / V_i ~ chi2(N). Caveat: sensitive to non-normality and has low power when T is small."""
    g = fe_resid.groupby(level=0)
    Ti = g.size().to_numpy(float)
    s2 = float((fe_resid ** 2).sum() / (Ti.sum() - len(Ti) - k))
    si2 = g.apply(lambda e: float((e ** 2).sum() / len(e))).to_numpy()
    Vi = g.apply(lambda e: float(((e ** 2 - (e ** 2).mean()) ** 2).sum() / (len(e) * (len(e) - 1)))).to_numpy()
    mask = Vi > 0
    W = float((((si2 - s2) ** 2)[mask] / Vi[mask]).sum())
    return {"W": W, "df": int(mask.sum()), "p": float(stats.chi2.sf(W, mask.sum()))}


def pesaran_cd(resid: pd.Series, min_overlap: int = 3) -> dict:
    """Pesaran (2004) CD = sqrt(2/(N(N-1))) * sum_{i<j} sqrt(T_ij) rho_ij  ~ N(0,1) (valid for small T, large N).
    Note: if the model has NO time effects, common macro shocks mechanically cause CD to reject."""
    W = resid.unstack(0)                      # rows = time, columns = banks
    N = W.shape[1]
    num, cnt = 0.0, 0
    cols = W.columns
    for a in range(N):
        for b in range(a + 1, N):
            pair = W.iloc[:, [a, b]].dropna()
            if len(pair) >= min_overlap and pair.std().min() > 0:
                num += np.sqrt(len(pair)) * np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]
                cnt += 1
    cd = np.sqrt(2.0 / (N * (N - 1))) * num
    return {"CD": float(cd), "p": float(2 * stats.norm.sf(abs(cd))), "pairs_used": cnt}


def vif_table(panel: pd.DataFrame, xs: list[str]) -> pd.Series:
    X = sm.add_constant(panel[xs], has_constant="add")
    return pd.Series([variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])], index=xs, name="VIF")


# --------------------------------------------------------------------------- #
# 5. DGP with known true parameters
# --------------------------------------------------------------------------- #
GDP = dict(zip(range(2013, 2024), [5.4, 6.0, 6.7, 6.2, 6.8, 7.1, 7.0, 2.9, 2.6, 8.0, 5.0]))   # illustrative (approximate)
INF = dict(zip(range(2013, 2024), [6.6, 4.1, 0.6, 2.7, 3.5, 3.5, 2.8, 3.2, 1.8, 3.2, 3.3]))   # illustrative (approximate)
TRUE_BETA = {"CAP": 0.11, "LDR": 0.020, "NPL": -0.055, "GDP": 0.03, "COVID": 0.20}


def simulate_bank_panel(n: int = 27, years=range(2013, 2024), seed: int = 0, endog_alpha: float = 0.0,
                        common_shock_sd: float = 0.15, rho: float = 0.0) -> pd.DataFrame:
    """NIM_it = a + b'X_it + alpha_i + u_t + e_it, with e_it AR(1) with coefficient rho.
    endog_alpha > 0: alpha_i is correlated with CAP → FE is consistent, RE is biased (Hausman should reject).
    COVID is coded CORRECTLY as {0,1} (=1 for 2020–2022). u_t = common year shock (source of cross-sectional dependence)."""
    rng = np.random.default_rng(seed)
    years = list(years)
    ent = np.repeat(np.arange(n), len(years))
    yr = np.tile(years, n)
    cap_i = rng.normal(8.7, 2.5, n)
    alpha = endog_alpha * (cap_i - cap_i.mean()) + rng.normal(0, 0.6, n)
    u_t = dict(zip(years, rng.normal(0, common_shock_sd, len(years))))
    cap = np.clip(cap_i[ent] + rng.normal(0, 0.9, n * len(years)), 3, None)
    ldr = np.clip(rng.normal(89, 12, n * len(years)), 40, 140)
    npl = np.clip(rng.lognormal(0.55, 0.55, n * len(years)), 0.3, 30)
    gdp = np.array([GDP[t] for t in yr]); inf = np.array([INF[t] for t in yr])
    covid = np.isin(yr, [2020, 2021, 2022]).astype(float)
    e = np.zeros(n * len(years))
    innov = rng.normal(0, 0.30, n * len(years))
    for k in range(len(e)):
        e[k] = innov[k] + (rho * e[k - 1] if (k % len(years)) > 0 else 0.0)
    nim = (0.8 + TRUE_BETA["CAP"] * cap + TRUE_BETA["LDR"] * ldr + TRUE_BETA["NPL"] * npl
           + TRUE_BETA["GDP"] * gdp + TRUE_BETA["COVID"] * covid + alpha[ent]
           + np.array([u_t[t] for t in yr]) + e)
    return pd.DataFrame(dict(bank=ent, year=yr, NIM=nim, CAP=cap, LDR=ldr, NPL=npl, INF=inf, GDP=gdp, COVID=covid))


# --------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame, y="NIM", xs=("CAP", "LDR", "NPL", "INF", "GDP", "COVID"),
                 covid_years=(2020, 2021, 2022)) -> dict:
    xs = list(xs)
    P = make_panel(df, "bank", "year", [y] + xs)
    out = {}
    out["covid_audit"] = audit_dummy(P, "COVID", expected_on=list(covid_years))
    out["macro_id"] = macro_identification(P, [c for c in ("INF", "GDP", "COVID") if c in xs])
    f = fit_models(P, y, xs)
    out["coefs"] = coef_table(f)
    firm_vars = [c for c in xs if c in ("CAP", "LDR", "NPL", "CAR")]
    out["F_poolability"] = {"stat": float(f.fe.f_pooled.stat), "p": float(f.fe.f_pooled.pval)}
    out["BP_LM"] = breusch_pagan_lm(f.pooled.resids)
    out["hausman_classic"] = hausman_classic(f, firm_vars)   # bank-level variables (FE–RE comparison is meaningful)
    out["mundlak"] = mundlak_test(P, y, xs)
    out["wooldridge_AR"] = wooldridge_ar(P, y, xs)
    out["modified_wald"] = modified_wald(f.fe.resids, k=len(xs))
    out["pesaran_CD_FE"] = pesaran_cd(f.fe.resids)
    out["pesaran_CD_TWFE"] = pesaran_cd(f.twfe.resids)
    out["VIF"] = vif_table(P, xs)
    return out


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    sim = simulate_bank_panel(endog_alpha=0.25, seed=1, rho=0.3)
    res = run_pipeline(sim)
    for k, v in res.items():
        print(f"--- {k}\n{v}")
    print("True coefficients:", TRUE_BETA)
