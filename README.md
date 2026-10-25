# Quantitative Finance & Econometrics Portfolio

**Le Nguyen Quoc Thang** — MSc student in Mathematical Economics; research interests: algorithmic trading, volatility forecasting, and data science for financial time series.

This repository re-examines my own coursework, thesis and student research with a "referee" mindset: each project was **re-estimated from the raw data, audited for data and specification errors, and corrected**. Where the original conclusion changes, the README says so.

| # | Project | Methods | Headline finding |
|---|---|---|---|
| 01 | [Bank NIM determinants, 27 banks 2013–23 (thesis)](01-bank-nim-thesis) | Panel FE/RE/TWFE, Mundlak, wild cluster bootstrap, Ho–Saunders, Nickell bias, rolling-origin forecasts | A mis-coded dummy flipped the COVID sign; year-level macro effects are not robust; the structural model loses to a naive forecast |
| 02 | [Bank NIM, short panel 2018–23](02-bank-nim-practicum) | Within/between decomposition, wild bootstrap with G = 6 | SIZE and CAP effects are between-bank, not within-bank |
| 03 | [Single-index model, 12 bank stocks vs VN30](03-single-index-model) | SIM, HAC betas, Dimson/Blume, Breusch–Pagan LM, two-index model, EGP, block bootstrap | The "return" series was a ratio of log prices, distorting betas; the EGP portfolio does not beat equal weight out of sample |
| 04 | [Sleep-disorder logit](04-sleep-disorder-logit) | Logit, separation, Firth, grouped CV, cluster-robust SE, multinomial logit | Naive CV overstates AUC (duplicate profiles); an omitted BMI variable drives the result |
| 05 | [Student research: copula, APARCH-DCC, spillovers](05-research-copula-volatility-spillover) | GJR-GARCH-copula, ARMA-APARCH/DCC, spillover index, wavelets | Frontier–US dependence is lower-tail and rises after COVID-19 |
| 06 | [Coursework methodology reviews](06-coursework-methodology-reviews) | VAR, ARIMA/GARCH, VaR/ES, binomial trees, Vasicek MLE | Reviews of specification, backtesting and benchmark issues |

## Reproducing
```bash
pip install -r requirements.txt
cd 01-bank-nim-thesis && python run.py --data <panel.xlsx> --out results
```
Raw data are **not** distributed (see each folder's *Data* section for what is required). Reusable diagnostics live in [`src/panelkit`](src/panelkit): panel model suite, Mundlak/Wooldridge/Pesaran tests, wild cluster bootstrap with Webb weights, Ho–Saunders two-step, Nickell-bias Monte Carlo, leave-one-bank-out and rolling-origin evaluation.

## Principles applied
Audit encodings and units before modelling; match inference to the true number of independent clusters; test whether a model beats a naive benchmark out of sample; report effect sizes next to p-values.

Licensed under MIT (code). Reports are my own analysis; no confidential or personal data are included.
