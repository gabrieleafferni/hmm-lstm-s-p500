# Hybrid HMM-LSTM model for multi-horizon S&P 500 return forecasting
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabriele-afferni/hmm-lstm-sp500/blob/main/hmm_lstm_pipeline.ipynb)
Probabilistic (quantile) forecasting of S&P 500 returns over 20-, 60- and 120-trading-day
horizons, combining a Hidden Markov Model that identifies market regimes with an LSTM that
predicts the full conditional distribution of future returns.

This repository contains the end-to-end pipeline developed for my MSc thesis in Statistics
(University of Turin), with results being written up for a peer-reviewed publication.

## Overview

Forecasting the direction of returns is hard; forecasting their distribution is more
useful and more honest. Instead of a single point prediction, the model outputs five
quantiles (5th, 25th, 50th, 75th, 95th), so the forecast comes with an explicit measure of
uncertainty and downside risk.

The approach has two stages:

- **Market regimes (GMM-HMM).** A 3-state Gaussian-mixture Hidden Markov Model is fitted to
  macro-financial features and identifies latent regimes (broadly: calm/bull, neutral,
  crisis/bear). For each day it produces the posterior probability of each regime.
- **Quantile LSTM.** An LSTM consumes the macro-financial features together with the regime
  posteriors (used as soft, continuous inputs rather than hard labels) and predicts the
  return quantiles via the pinball (quantile) loss.

## Method highlights

- **Walk-forward cross-validation** with a two-stage hyperparameter search (random search,
  then refinement on the top configurations) to avoid look-ahead bias.
- **Multi-seed ensembling** of the final model to reduce the variance of neural-network
  training.
- **Soft regime inputs:** the HMM posterior probabilities are fed to the LSTM as continuous
  features rather than hard (Viterbi) state labels, so that the uncertainty present during
  regime transitions is preserved instead of being discretised away.
- **Rigorous, sceptical evaluation:** calibration coverage (PICP), Winkler scores and formal
  statistical tests — Kupiec, Christoffersen, Berkowitz and Diebold-Mariano — rather than a
  single headline metric.
- **Honest baselines and ablation:** the model is compared against historical-quantile and
  GARCH(1,1) (normal and Student-t) benchmarks, and an ablation study removes the HMM
  features to isolate their contribution.

## Results (summary)

On the pinball (quantile) loss, the hybrid model achieves a lower error than the
historical-quantile and GARCH(1,1) baselines at all three horizons. Diebold-Mariano tests —
computed against the historical-quantile and GARCH-t benchmarks — support the improvement
over these baselines.

Two honest caveats are worth stating up front:

- The GARCH benchmark is fitted on overlapping aggregated horizon returns, which makes its
  predictive intervals very narrow (its 90% intervals cover only a small fraction of
  outcomes). The large pinball gap against GARCH is inflated by this mis-scaling and should
  be read with caution; the comparison against the historical-quantile benchmark is the more
  informative one.
- Interval coverage is close to nominal at the 20- and 60-day horizons and degrades at 120
  days, where heavily overlapping targets sharply reduce the effective sample size.
  Calibration is therefore assessed *per horizon* (PICP, Winkler, and the
  Kupiec/Christoffersen/Berkowitz tests) rather than claimed globally.

The most informative result comes from the **ablation study**, and it is strongly
horizon-dependent. Removing the HMM regime posteriors is significantly *better* at 20 days
(regime information hurts short-horizon forecasts; DM p ≈ 0.008), makes no significant
difference at 60 days (p ≈ 0.73), and is significantly *worse* at 120 days (regime
information helps; p ≈ 0.044). In short, the regime signal is beneficial only at the longest
horizon and is actively detrimental at the shortest — consistent with regimes being a
slow-moving signal that adds noise rather than information over short windows.

The small summary tables behind these statements are in `results/`.

## Repository structure

```
.
├── hmm_lstm_pipeline.ipynb   # full end-to-end pipeline (22 numbered stages)
├── requirements.txt          # Python dependencies
├── results/                  # small summary tables and selected figures
└── README.md
```

The notebook is organised into numbered stages: data acquisition and preprocessing,
HMM feature preparation, diagnostics, GMM-HMM fitting, regime visualisation and diagnostics,
the quantile-LSTM pipeline (CV, refinement, multi-seed ensemble), calibration metrics,
baselines, formal tests, regime-conditional analysis, ablation and economic evaluation.

## How to run

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Set a free FRED API key (https://fred.stlouisfed.org/)
export FRED_API_KEY="your_key_here"

# 3. (optional) choose the output folder; defaults to ./outputs
export HMM_LSTM_DIR="outputs"

# 4. Launch the notebook
jupyter notebook hmm_lstm_pipeline.ipynb
```

Run the cells top to bottom. The first two stages set up the output folder and (optionally)
install the dependencies, so the notebook is also self-contained on a fresh environment or
on Google Colab. All intermediate artefacts (datasets, HMM states/posteriors, predictions,
metrics) are written as CSV files in the output folder and reused by later stages.

> **Note.** A GPU is recommended for the LSTM stages but not required. The full hyperparameter
> search over all three horizons can take a while on CPU.

## Data sources

- **Market data** (S&P 500, VIX, VIX3M, oil, gold, US dollar index): Yahoo Finance, accessed
  via `yfinance`.
- **Macroeconomic data** (e.g. yield differentials): FRED, via the `fredapi` client.

Raw data is not redistributed in this repository; it is downloaded at runtime from the
sources above. A free `FRED_API_KEY` is required, because the yield-spread feature is part of
the model.

## Author

Gabriele Afferni — MSc in Statistics, University of Turin

LinkedIn · GitHub
