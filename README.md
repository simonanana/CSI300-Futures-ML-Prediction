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
CSI300-Futures-ML-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── csi300_ml_prediction.ipynb              # Main notebook (all three tasks)
│
├── data/
│   ├── README.md                           # Data dictionary and sourcing instructions
│   │
│   ├── descriptive_statistics/             # Exploratory data analysis
│   │   ├── feature_correlation_matrix.png
│   │   ├── feature_importance_plot.png
│   │   ├── regime_performance_plot.png
│   │   │
│   │   ├── CSI300_macro/                   # Macroeconomic descriptive statistics
│   │   │   ├── macro_stats.py
│   │   │   ├── macro_correlation_heatmap.png
│   │   │   ├── macro_boxplot_plots.png
│   │   │   ├── macro_hist_plots.png
│   │   │   ├── macro_kde_plots.png
│   │   │   └── macro_*_timeseries.png      # Time-series plots for macro variables
│   │   │
│   │   └── CSI300_micro/                   # Micro-level (futures contract) descriptive statistics
│   │       ├── micro_stats.py
│   │       ├── micro_data_cleaned.csv
│   │       ├── csi300_correlation_heatmap.png
│   │       ├── csi300_boxplot_plots.png
│   │       ├── csi300_hist_plots.png
│   │       ├── csi300_kde_plots.png
│   │       ├── csi300_time_series.png
│   │       ├── numerical_boxplot_plots.png
│   │       └── numerical_density_plots.png
│   │
│   ├── direction/                          # Task 1 data
│   │   ├── sample_direction.csv            # Sample data (first 20 rows)
│   │   ├── train15-23.xlsx                 # Training set: 2015–2023
│   │   └── test24-25.xlsx                  # Testing set: 2024–2025
│   │
│   ├── volatility/                         # Task 2 data
│   │   ├── data_train_2015_2023.xlsx       # Training set: 2015–2023
│   │   ├── data_test_2024_2025.xlsx        # Testing set: 2024–2025
│   │   └── data_cleaned_no_leakage.csv     # Cleaned data with leakage features removed
│   │
│   └── sentiment/                          # Task 3 data
│       └── Sentiment_Target_Enhanced_data.csv
│
└── output/
    ├── direction/                          # Task 1 results
    │   ├── traditional_ml_classification_results.csv
    │   ├── deep_learning_classification_results.csv
    │   ├── trading_strategy_metrics.csv
    │   ├── model_comparison_results.png
    │   ├── roc_curves_comparison.png
    │   ├── shap_classification_summary.png
    │   ├── shap_classification_importance.png
    │   └── comprehensive_strategy_analysis.png
    │
    ├── volatility/                         # Task 2 results
    │   ├── model_comparison_visualization.png
    │   ├── volatility_models_comparison_visualization.png
    │   ├── regime_analysis_visualization.png
    │   ├── time_series_prediction.png
    │   └── feature_engineering_summary.png
    │
    └── sentiment/                          # Task 3 results
        ├── complete_analysis_results_EN.xlsx
        ├── model_comparison.png
        ├── model_comparison_complete_EN.png
        ├── roc_curves_all_models.png
        ├── comprehensive_analysis.png
        ├── comprehensive_strategy_analysis.png
        ├── strategy_analysis_complete_EN.png
        ├── threshold_analysis.png
        └── metric_comparison_by_threshold.png
``````

> **Note:** Raw datasets are included for reproducibility. The `data/` directory also contains a `descriptive_statistics/` subdirectory with EDA scripts and visualizations characterizing the dataset at both macro and micro levels. See [`data/README.md`](data/README.md) for column definitions, descriptive statistics documentation, and data sourcing instructions.

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
