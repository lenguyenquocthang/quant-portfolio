# Quantitative Finance & Econometrics Portfolio

**Le Nguyen Quoc Thang** — MSc student in Mathematical Economics; research interests: algorithmic trading, volatility forecasting, and data science for financial time series.

Each project below re-examines my own coursework, thesis or student research with a "referee" mindset: **re-estimated from the raw data, audited for data and specification errors, and corrected**. Where the original conclusion changes, the README says so. Each project lives in its own repository.

| Project | Methods | Headline finding |
|---|---|---|
| [bank-nim-determinants](https://github.com/lenguyenquocthang/bank-nim-determinants) | Panel FE/RE/TWFE, Mundlak, wild cluster bootstrap, Ho–Saunders, Nickell bias, rolling-origin forecasts | A mis-coded dummy flipped the COVID sign; year-level macro effects are not robust; the structural model loses to a naive forecast |
| [bank-nim-practicum](https://github.com/lenguyenquocthang/bank-nim-practicum) | Within/between decomposition, wild bootstrap with G = 6 | SIZE and CAP effects are between-bank, not within-bank |
| [single-index-model-vn-banks](https://github.com/lenguyenquocthang/single-index-model-vn-banks) | SIM, HAC betas, Dimson/Blume, Breusch–Pagan LM, two-index model, EGP, block bootstrap | The "return" series was a ratio of log prices, distorting betas; EGP does not beat equal weight out of sample |
| [sleep-disorder-logit](https://github.com/lenguyenquocthang/sleep-disorder-logit) | Logit, separation, Firth, grouped CV, cluster-robust SE, multinomial logit | Naive CV overstates AUC (duplicate profiles); an omitted BMI variable drives the result |
| [frontier-markets-copula-research](https://github.com/lenguyenquocthang/frontier-markets-copula-research) | GJR-GARCH-copula, ARMA-APARCH/DCC, spillover index, wavelets | Frontier–US dependence is lower-tail and rises after COVID-19 |
| [quant-coursework-reviews](https://github.com/lenguyenquocthang/quant-coursework-reviews) | VAR, ARIMA/GARCH, VaR/ES, binomial trees, Vasicek MLE | Reviews of specification, backtesting and benchmark issues |

Raw data are not distributed; each repo's README says what is needed. The reusable panel diagnostics (Mundlak/Wooldridge/Pesaran tests, wild cluster bootstrap, Ho–Saunders, Nickell Monte Carlo, rolling-origin evaluation) are in `src/panelkit` of the two bank-NIM repos.

Principles: audit encodings and units before modelling; match inference to the true number of independent clusters; test whether a model beats a naive benchmark out of sample; report effect sizes next to p-values.
