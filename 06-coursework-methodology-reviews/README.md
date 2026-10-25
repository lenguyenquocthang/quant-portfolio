# 06 · Coursework methodology reviews (quantitative courses)

Short, English methodological reviews of coursework papers: what is correct, what is wrong, and how I would redo the analysis. No raw data or report text is included; where papers were group work these notes review the method only. Each review reproduces the numbers from the paper's own tables where possible.

## Time-series analysis — VAR for US exports, imports, inflation and GDP growth (1970–2021, n = 52)
* ADF/KPSS: EX, IM and ICP are I(1), GDPG is I(0); VAR(1) in (ΔEX, GDPG, ΔICP, ΔIM) chosen by AIC/HQ/FPE. Reproduced the Granger p-values and the 10-year FEVD to the digit under the same Cholesky order.
* Issues: ADF specification not stated (reported p-values differ from a BIC-lag constant-only test); inflation is a rate, so differencing risks over-differencing (regime shift 1973–82); EX and IM as %-of-GDP share a denominator and EX is a GDP component (accounting identity); 12 Granger tests without multiplicity control (one p = 0.049 is borderline); IRFs reported without bands; heteroskedasticity test (White p = 0.022) interpreted at inconsistent levels; cointegration ignored.
* Forecast: the 2022 VAR forecast is not reproducible; the model has no out-of-sample edge (expanding-window RMSE ratio vs historical mean 0.99 for GDPG, 1.14 for ΔEX). Suitable for description, not forecasting.

## International financial models — ARIMA vs ARCH-GARCH for weekly EUR/USD (2018–2024)
* Conclusion "ARIMA is more accurate" contradicts the paper's own table (GARCH has lower absolute error in 3 of 5 weeks; MAE 0.31% vs 0.36%).
* Issues: no random-walk benchmark (adjusted R² 0.013–0.035 → near random walk); isolated AR(19) lag selected from a 20-lag scan (≈64% chance of a spurious PACF spike); AIC compared across different effective samples; ARCH-LM does not reject but GARCH is fitted anyway; "forecast" is static one-step on observed prices rather than true out-of-sample; no Diebold–Mariano / Clark–West test.
* Redo: log returns, daily data, random-walk benchmark, expanding-window forecasts, Theil-U and DM(HLN) for prices, QLIKE + Mincer–Zarnowitz for volatility.

## Financial risk models — VaR/ES for five large Vietnamese bank stocks (2020–2023)
* Reported descriptive statistics violate arithmetic bounds for n = 207 (|skewness| ≤ √n ≈ 14.4), so skewness/kurtosis were mis-computed; variance reported in inconsistent units.
* Parametric VaR in the table decreases with confidence level (impossible; VaR = zσ must increase) and ES equals VaR at 90%; the conclusion that "99% gives the highest return" inverts the meaning of VaR.
* Cumulative "5/10/15-day vs base date" returns overlap and are not i.i.d.; HS 99% VaR rests on ~2 observations; no Kupiec/Christoffersen backtests (Kupiec recomputed from the paper's counts: LR ≈ 3.56, p ≈ 0.059).
* Redo: filtered historical simulation with GARCH-t, multi-day VaR by simulation, walk-forward backtests with Kupiec, Christoffersen and DQ.

## Financial asset pricing 1 — factor models, bonds, binomial tree, VaR for three equities
* Correct and kept: factor-model variance decomposition, bond duration/convexity, replicating portfolios.
* Binomial tree for a single stock mixes **log returns with simple returns** (E[S<sub>n</sub>] should be S₀(pu+(1−p)d)<sup>n</sup>, not S₀(1+r̄)); CRR parameters mis-stated (u = e<sup>σ√Δt</sup>, risk-neutral q = (e<sup>rΔt</sup>−d)/(u−d)).
* Historical VaR at 99% sits at the ±7% daily price limit of the exchange (truncated tail); summing stock VaRs ignores diversification; no backtest or bootstrap CI.

## Mathematical finance — Vasicek interest-rate model by MLE (1967–2014, 48 obs)
* Reproduced the MLE (b = 0.011897, ln L = 104.5285) to six digits, but the reported "a" and "σ" are the **discrete AR(1)** parameters (a = 1 − φ = 0.2931, σ<sub>ε</sub> = 0.0262). Continuous-time Vasicek parameters are a = −ln φ̂ = 0.3469 and σ = σ<sub>ε</sub>√(2a/(1−φ²)) = 0.0308, which must be used in A(τ), B(τ) and in the long-run rate b − σ²/(2a²).
* Suggested upgrades: standard errors from the numerical Hessian, a unit-root check on φ, comparison with CIR and Hull–White, and risk-neutral calibration to a market curve.
