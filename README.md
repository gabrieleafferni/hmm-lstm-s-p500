Hybrid HMM-LSTM model for multi-horizon S&P 500 return forecasting
Probabilistic (quantile) forecasting of S&P 500 returns over 20-, 60- and 120-trading-day
horizons, combining a Hidden Markov Model that identifies market regimes with an LSTM that
predicts the full conditional distribution of future returns.
This repository contains the end-to-end pipeline developed for my MSc thesis in Statistics
(University of Turin), with results being written up for a peer-reviewed publication.
Overview
Forecasting the direction of returns is hard; forecasting their distribution is more
useful and more honest. Instead of a single point prediction, the model outputs five
quantiles (5th, 25th, 50th, 75th, 95th), so the forecast comes with an explicit measure of
uncertainty and downside risk.
The approach has two stages:
Market regimes (GMM-HMM). A 3-state Gaussian-mixture Hidden Markov Model is fitted to
macro-financial features and identifies latent regimes (broadly: calm/bull, neutral,
crisis/bear). For each day it produces the posterior probability of each regime.
Quantile LSTM. An LSTM consumes the macro-financial features together with the regime
posteriors (used as soft, continuous inputs rather than hard labels) and predicts the
return quantiles via the pinball (quantile) loss.
Method highlights
Walk-forward cross-validation with a two-stage hyperparameter search (random search,
then refinement on the top configurations) to avoid look-ahead bias.
Multi-seed ensembling of the final model to reduce the variance of neural-network
training.
Soft regime inputs: the HMM posterior probabilities are fed to the LSTM as continuous
features, which is more informative than discrete (Viterbi) state labels during regime
transitions.
Rigorous, sceptical evaluation: calibration coverage (PICP), Winkler scores and formal
statistical tests — Kupiec, Christoffersen, Berkowitz and Diebold-Mariano — rather than a
single headline metric.
Honest baselines and ablation: the model is compared against historical-quantile and
GARCH(1,1) (normal and Student-t) benchmarks, and an ablation study removes the HMM
features to isolate their contribution.
Results (summary)
The hybrid model produces well-calibrated quantile forecasts and systematically
outperforms the standard benchmarks used in the field on the pinball loss, with the
clearest and most statistically significant gains against the GARCH benchmarks across all
horizons (Diebold-Mariano test).
The ablation study indicates that the HMM regime information contributes mainly at the
longer horizons, which is consistent with regimes being a slow-moving signal.
The small summary tables behind these statements are in `results/`.
Repository structure
```
.
├── hmm\_lstm\_pipeline.ipynb   # full end-to-end pipeline (22 stages)
├── requirements.txt          # Python dependencies
├── results/                  # small summary tables and selected figures
└── README.md
```
The notebook is organised into numbered stages: data acquisition and preprocessing,
HMM feature preparation, diagnostics, GMM-HMM fitting, regime visualisation and diagnostics,
the quantile-LSTM pipeline (CV, refinement, multi-seed ensemble), calibration metrics,
baselines, formal tests, regime-conditional analysis, ablation and economic evaluation.
How to run
```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Set a free FRED API key (https://fred.stlouisfed.org/)
export FRED\_API\_KEY="your\_key\_here"

# 3. (optional) choose the output folder; defaults to ./outputs
export HMM\_LSTM\_DIR="outputs"

# 4. Launch the notebook
jupyter notebook hmm\_lstm\_pipeline.ipynb
```
All intermediate artefacts (datasets, HMM states/posteriors, predictions, metrics) are
written as CSV files in the output folder and reused by later stages, so the notebook can be
run top to bottom.
> \*\*Note.\*\* A GPU is recommended for the LSTM stages but not required. Full hyperparameter
> search over all three horizons can take a while on CPU.
Data sources
Market data (S&P 500, VIX, VIX3M, oil, gold, US dollar index): Yahoo Finance, accessed
via `yfinance`.
Macroeconomic data (e.g. yield differentials): FRED via
the `fredapi` client.
Raw data is not redistributed in this repository; it is downloaded at runtime from the
sources above.
Author
Gabriele Afferni — MSc in Statistics, University of Turin
LinkedIn · GitHub
