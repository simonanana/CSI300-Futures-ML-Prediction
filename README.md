# CSI 300 Futures Prediction: A Multi-Task Machine Learning Benchmark

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Benchmarking 10 ML/DL models across three prediction tasks on China's CSI 300 Index Futures — direction, volatility, and sentiment breakthrough — with walk-forward evaluation and trading strategy analysis.**

## Overview

This project provides a systematic comparison of machine learning and deep learning approaches for forecasting the CSI 300 Index Futures, one of China's most actively traded derivatives. We formulate three distinct prediction tasks and evaluate a broad model suite on each, using a strict temporal train-test split to ensure realistic out-of-sample performance estimates.

| Task | Type | Models | Best OOS Metric |
|------|------|--------|-----------------|
| Price Direction | Binary Classification | 10 (ML + DL) | 52.1% Accuracy / 0.559 AUC |
| Realized Volatility | Regression | 10 | R² = 0.900 (Linear Reg.) |
| Sentiment Breakthrough | Binary Classification + Trading | 5 | AUC = 0.719 / −1.3% Max DD |

**Training Period:** 2015–2023 &nbsp;|&nbsp; **Testing Period:** 2024–2025 (True Out-of-Sample)

## Key Results

### Task 1 — Price Direction Prediction

Ten models were evaluated, including traditional classifiers (Decision Tree, SVM, Logistic Regression, Gradient Boosting, LightGBM) and deep learning architectures (LSTM, Bidirectional LSTM, LSTM + Attention, 1D-CNN).

All models achieve near-random test accuracy (~50–52%), confirming the well-documented weak-form efficiency of index futures markets at the daily frequency. The substantial train-test accuracy gap observed in ensemble methods (e.g., Gradient Boosting: 96.9% train → 51.6% test) highlights the overfitting risk inherent in high-capacity models applied to noisy financial time series.

### Task 2 — Volatility Prediction

Linear and Ridge regression achieve the highest out-of-sample R² (~0.90) with minimal overfitting, outperforming complex ensemble methods. This result is consistent with volatility's well-known autoregressive structure — simple linear models effectively capture the dominant mean-reverting signal without overfitting to noise.

XGBoost achieves the highest directional accuracy (72.2%), suggesting potential value for volatility regime detection despite lower R².

### Task 3 — Sentiment Breakthrough Prediction

We construct a composite sentiment indicator from RSI, KDJ, CCI, and Williams %R, and predict next-day sentiment breakthroughs (change > threshold). Using Random Forest predictions with a 0.6 probability threshold:

- **93% reduction in maximum drawdown** (−1.30% vs. −18.98% benchmark)
- **Sharpe Ratio: 0.90** (vs. benchmark 0.93)
- **Hit Rate: 78.3%** across 27 trades over ~2 years
- **Signal Stability:** 14.9-day average holding period

This demonstrates the practical value of sentiment breakthrough prediction as a risk management tool for selective market entry.

## Repository Structure

```
csi300-futures-ml-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── csi300_ml_prediction.ipynb          # Main notebook — all three tasks
│
├── data/
│   ├── README.md                       # Data dictionary and sourcing instructions
│   ├── direction/
│   │   ├── sample_direction.csv        # Sample data (first 20 rows)
│   │   ├── train_2015_2023.xlsx        # Full training set (not included)
│   │   └── test_2024_2025.xlsx         # Full testing set (not included)
│   ├── volatility/
│   │   ├── sample_volatility.csv
│   │   ├── train_2015_2023.csv
│   │   └── test_2024_2025.csv
│   └── sentiment/
│       ├── sample_sentiment.csv
│       └── sentiment_training_data.csv
│
└── output/
├── direction/
│   ├── direction_prediction_results.csv
│   ├── roc_curves_comparison.png
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── trading_strategy_comparison.png
│   └── strategy_comparison_metrics.csv
├── volatility/
│   ├── volatility_prediction_results.csv
│   ├── model_comparison_visualization.png
│   ├── regime_analysis_results.csv
│   ├── regime_analysis_visualization.png
│   ├── time_series_prediction.png
│   └── feature_importance.csv
└── sentiment/
├── sentiment_classification_results.csv
├── sentiment_strategy_results.csv
└── sentiment_backtest_visualization.png
```

## Models

### Classification (Direction & Sentiment)

| Category | Models |
|----------|--------|
| Linear | Logistic Regression (L1/L2) |
| Tree-Based | Decision Tree, Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost |
| Kernel | SVM (Linear), SVM (RBF) |
| Deep Learning | Simple LSTM, Bidirectional LSTM, LSTM + Attention, 1D-CNN |

### Regression (Volatility)

| Category | Models |
|----------|--------|
| Linear | Linear Regression, Ridge, Lasso, ElasticNet |
| Tree-Based | Decision Tree, Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost |

## Methodology

- **Feature engineering:** Technical indicators including momentum, trend, volatility, and volume-based features. Composite sentiment score constructed from RSI, KDJ, CCI, and Williams %R.
- **Data leakage prevention:** Strict exclusion of look-ahead features; temporal train/test split with no shuffling.
- **Evaluation:** Multiple metrics per task (AUC, F1, R², RMSE, Direction Accuracy). SHAP values for model interpretability.
- **Regime analysis:** Quantile-based market regime identification (Low / Medium / High volatility) with per-regime performance evaluation.
- **Trading backtesting:** Five strategy variants (Simple Signal, High Confidence, Long-Short, Low Volatility Filter, Dynamic Position) with Sharpe ratio, maximum drawdown, Calmar ratio, and win rate.

## Requirements

```
python >= 3.8
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
tensorflow >= 2.x
shap
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost tensorflow shap matplotlib seaborn openpyxl
```

## Usage

1. Place your CSI 300 futures data files in the `data/` directory (see structure above).
2. Open and run `csi300_ml_prediction.ipynb` sequentially.
3. Results and plots will be saved to `output/`.

> **Note:** Data files are not included in this repository due to licensing restrictions.

