# 05 · Student research: dependence, volatility and spillovers

Three quantitative research reports written as part of a student research team (2024). The write-ups below summarise design and findings; the full reports, raw price data and estimation scripts are not included in this repository. My role: co-author in a team of four; the repo contains no group member's personal data.

## 1. Dependence structure between frontier markets and the US stock market — GJR-GARCH-copula
* **Data:** daily index returns, 7 Jan 2014 – 27 Mar 2024. US vs Vietnam, Argentina, Croatia, Morocco, Oman.
* **Method:** marginals by GJR-GARCH (selected by information criteria; goodness-of-fit by Anderson–Darling and Kolmogorov–Smirnov tests); pair copulas from the elliptical (Gaussian, t), Archimedean (Clayton, Gumbel, Frank, Joe, rotated versions) and mixed (BB1, BB6, BB7, BB8) families, chosen by information criteria; two-step inference-functions-for-margins (IFM) estimation; sub-sample comparison before vs after COVID-19.
* **Findings:** in calm markets dependence is positive but low (Kendall-type dependence below 0.2, symmetric tail dependence below 0.1). In stress, dependence is concentrated in the **lower tail** — i.e. frontier markets co-move with the US mainly when the US falls — and is material for Croatia, Vietnam and Argentina. Before COVID-19 tail dependence is symmetric; **after COVID-19 lower-tail dependence rises significantly**.
* **Implication:** diversification benefits from frontier markets are weakest exactly when they are needed.

## 2. Asymmetric volatility and risk spillovers from gold, WTI crude and Bitcoin to the Vietnamese stock market — ARMA-APARCH / DCC-APARCH
* **Data:** daily returns, 2018–2024.
* **Method:** ARMA(1,1)-APARCH(1,1) marginals for asymmetry/leverage; DCC-APARCH for dynamic conditional correlations.
* **Findings:** Vietnamese equities and Bitcoin react more strongly to negative shocks; gold reacts more to positive shocks; WTI volatility is nearly symmetric. Correlations of WTI and Bitcoin with Vietnamese equities are low and positive but **rise during COVID-19** (contagion), while gold's correlation is negative (safe-haven behaviour).

## 3. Spillovers from world equities, crude oil and gold to the domestic SJC gold price — Spillover Index and Wavelet Coherence
* **Data:** daily, 3 Jan 2018 – 29 Dec 2023.
* **Method:** Spillover Index of the Diebold–Yilmaz type and wavelet coherence (time–frequency co-movement and lead–lag).
* **Findings:** SJC gold is driven by world gold, but the effect **declines after COVID-19 and during the Russia–Ukraine war** as the domestic price decouples from the world price (policy-induced premium). WTI affects SJC gold positively before and after COVID. SJC gold has very low correlation with Vietnamese equities, but in 2020 gold shocks affected equities negatively, supporting gold as a hedge in stress.

## Limitations (author's own assessment)
Estimates are in-sample with daily non-synchronous trading hours across markets (a source of spurious lead–lag); the number of copula and marginal candidates searched is large relative to inference, so selection uncertainty is not reflected in the standard errors.
