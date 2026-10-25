# 01 · Determinants of bank net interest margin (27 Vietnamese banks, 2013–2023)

Corrected re-estimation of an undergraduate thesis. The original reported FGLS (`xtgls`) results and a COVID-19 effect; this folder rebuilds the pipeline from the raw workbook and audits the data, the identification of the macro variables and the inference.

**Panel:** 27 banks × 11 years = 297 obs. **Dependent variable:** NIM. **Bank-level regressors:** CAR, CAP, LDR, NPL. **Macro regressors:** ER, INF, GDP, COVID-19 dummy.

## Data
Not included. The workbook (sheet `Full`) was assembled from bank annual reports; replace with your own panel with the same columns (`ID, Year, NIM, CAR, CAP, LDR, NPL, ER, INF, GDP, COVID-19`).

```bash
pip install -r ../requirements.txt
python run.py --data path/to/panel.xlsx --out results   # ~10 s
```
Outputs: `results/kltn_results.json` (all tables/tests) and `results/figures/*.png`.

## What was wrong, and what changed
| Issue | Finding |
|---|---|
| Text dummy `Có/Không` encoded alphabetically (Có=1, Không=2) | COVID sign **inverted**. Corrected FE coefficient on COVID is **+0.187** (bank-clustered p = 0.002). |
| Macro variables are year-level (only T = 11 distinct values) | Bank-clustered SE overstate precision. Wild cluster bootstrap by year (Webb weights, G = 11): GDP p = **0.038**, COVID p = **0.32**, INF p = 0.20, ER p = 0.51. A Ho–Saunders two-step makes GDP insignificant (p = 0.59 in the version with COVID). |
| `xtgls` with AR(1) is a pooled GLS with **no bank effects** | F-test for poolability = 26.5 (p < 0.001), Breusch–Pagan LM = 657: pooled/GLS is rejected in favour of bank effects. |
| Hausman vs. cluster-robust inference | Mundlak (cluster-robust) p = 0.48, so RE is not rejected, but FE is retained for bank effects correlated with regressors. |
| Dynamics | Pooled ρ = 0.87 vs FE ρ = 0.46 brackets the true persistence (Nickell bias). Nickell-corrected ρ ≈ 0.70; Anderson–Hsiao ρ ≈ 0.43 with a weak instrument (first-stage F ≈ 3). GMM is unstable with N = 27. |

## Main FE results (bank-clustered SE)
| Variable | Coef. | p | Standardised effect (per 1 s.d., TWFE) |
|---|---|---|---|
| CAP | 0.108 | 0.004 | +0.34 |
| LDR | 0.024 | <0.001 | +0.38 |
| NPL | −0.057 | <0.001 | −0.12 |
| CAR | 0.007 | 0.80 | +0.02 |

## Robustness and predictive value
* Leave-one-bank-out: the NPL coefficient ranges from −0.082 (dropping NVB) to −0.051 (dropping STB), and the worst-case p-value is 0.138. Restricting to NPL < 10 gives −0.069 (p = 0.21). **The NPL result is sensitive to a few banks.**
* Rolling-origin out-of-sample RMSE (origin 2017): FE **0.597**, pooled 1.05, naive NIM<sub>t−1</sub> **0.464**. The structural model **does not beat a no-change forecast**; it explains cross-sectional differences, not next-year NIM.
* Specification curves for COVID and GDP across control sets are in the JSON (`spec_covid`, `spec_gdp`).

## Caveats
Small N (27 banks, 4 state-owned), T = 11 years, no instruments. Estimates are conditional associations, not causal effects.
