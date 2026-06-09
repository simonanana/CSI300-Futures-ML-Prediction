# Data Dictionary

This directory contains the datasets used across the three prediction tasks. **Full datasets are not included** in this repository due to licensing restrictions. Sample files (first 20 rows) are provided to illustrate the schema.

## Descriptive Statistics

The `descriptive_statistics/` directory contains exploratory data analysis (EDA) outputs that characterize the CSI 300 futures dataset prior to modeling. These are organized into two levels:

### Macro-Level Analysis (`CSI300_macro/`)

Descriptive statistics and visualizations for macroeconomic variables relevant to the CSI 300 Index, including the CSI 300 Index change rate, ChiNext Index, Shanghai Composite Index, China CPI (fixed base), and the 10-Year Treasury yield. Outputs include:

- `macro_stats.py` — Script to generate all macro-level descriptive statistics and plots.
- `macro_correlation_heatmap.png` — Pearson correlation matrix across macroeconomic variables.
- `macro_hist_plots.png` — Histograms showing the empirical distribution of each macro variable.
- `macro_kde_plots.png` — Kernel density estimates for distributional shape analysis.
- `macro_boxplot_plots.png` — Box plots highlighting medians, quartiles, and outliers.
- `macro_*_timeseries.png` — Time-series plots for selected variable pairs (e.g., CSI 300 vs. ChiNext change rate, CPI vs. 10Y Treasury yield).

### Micro-Level Analysis (`CSI300_micro/`)

Descriptive statistics and visualizations for the futures contract's own features — technical indicators, price-derived variables, and trading activity metrics. Outputs include:

- `micro_stats.py` — Script to generate all micro-level descriptive statistics and plots.
- `micro_data_cleaned.csv` — Cleaned micro-level feature dataset used in the analysis.
- `csi300_correlation_heatmap.png` — Feature correlation matrix for multicollinearity diagnostics.
- `csi300_hist_plots.png` / `csi300_kde_plots.png` — Distribution analysis of technical indicators.
- `csi300_boxplot_plots.png` / `numerical_boxplot_plots.png` — Outlier and spread analysis.
- `csi300_time_series.png` — Time-series visualization of key contract-level features.
- `numerical_density_plots.png` — Density plots for all numerical features.

### Cross-Cutting Plots

Three additional EDA plots sit at the `descriptive_statistics/` root level:

- `feature_correlation_matrix.png` — Full correlation matrix across all model input features.
- `feature_importance_plot.png` — Preliminary feature importance ranking (pre-modeling).
- `regime_performance_plot.png` — Visual comparison of model behavior across volatility regimes.

## Data Source

All data are derived from the **CSI 300 Index Futures (IF)** continuous main contract, sourced from [Wind](https://www.wind.com.cn/) / [CSMAR](https://www.gtarsc.com/) / [Tushare](https://tushare.pro/). Daily frequency.

To reproduce, obtain daily OHLC prices and pre-computed technical indicators for the CSI 300 Index Futures from any of the above providers, covering 2015-01-01 through 2025-04-30.


## Task 1 — Direction Prediction

**Directory:** `data/direction/`

| File | Period | Format | Rows (approx.) |
|------|--------|--------|-----------------|
| `train_2015_2023.xlsx` | 2015-01 to 2023-12 | Excel (.xlsx) | ~2,180 |
| `test_2024_2025.xlsx` | 2024-01 to 2025-04 | Excel (.xlsx) | ~320 |

### Column Definitions

| Column | English Name | Type | Description |
|--------|-------------|------|-------------|
| `代码` | code | str | Futures contract code |
| `名称` | name | str | Contract name |
| `日期` | date | date | Trading date (YYYY-MM-DD) |
| `收盘价(元)` | close_price | float | Daily closing price (CNY) |
| `开盘价(元)` | open_price | float | Daily opening price (CNY) |
| `最高价(元)` | high_price | float | Daily high price (CNY) |
| `最低价(元)` | low_price | float | Daily low price (CNY) |
| `结算价` | settlement_price | float | Daily settlement price (CNY) |
| `涨跌幅` | pct_change | float | Daily percentage change (%) |
| `RSI1` | RSI_14 | float | Relative Strength Index (14-day) |
| `K` | KDJ_K | float | KDJ indicator — K value |
| `D` | KDJ_D | float | KDJ indicator — D value |
| `J` | KDJ_J | float | KDJ indicator — J value |
| `CCI` | CCI | float | Commodity Channel Index |
| `W&R` | Williams_R | float | Williams %R |
| `MACD` | MACD | float | Moving Average Convergence Divergence |
| `DIF` | MACD_DIF | float | MACD signal line (DIF) |
| `DEA` | MACD_DEA | float | MACD signal line (DEA) |
| ... | ... | float | Additional technical indicators |

> **Note:** The target variable `direction` is generated at runtime: `direction = 1` if next-day return > 0, else `0`. It is not stored in the raw data files.


## Task 2 — Volatility Prediction

**Directory:** `data/volatility/`

| File | Period | Format | Rows (approx.) |
|------|--------|--------|-----------------|
| `train_2015_2023.csv` | 2015-01 to 2023-12 | CSV | ~2,180 |
| `test_2024_2025.csv` | 2024-01 to 2025-04 | CSV | ~320 |

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `日期` / `date` | date | Trading date |
| `volatility` | float | **Target variable** — realized volatility (annualized, based on intraday range) |
| `RSI1`, `K`, `D`, `J`, `CCI`, `W&R` | float | Technical oscillators (same as Task 1) |
| `MACD`, `DIF`, `DEA` | float | Trend indicators |
| Other technical features | float | ~40 pre-computed features |

### Excluded Columns (Data Leakage)

The following columns are present in the raw data but **must be excluded** during training to prevent target leakage:

| Column | Reason |
|--------|--------|
| `SD（标准差）` | Direct computation of the target |
| `volatility_percentile` | Derived from target |
| `high_volatility_regime` | Derived from target |
| `low_volatility_regime` | Derived from target |
| `volatility_target` | Future-looking label |


## Task 3 — Sentiment Breakthrough Prediction

**Directory:** `data/sentiment/`

| File | Period | Format | Rows (approx.) |
|------|--------|--------|-----------------|
| `Sentiment_Target_Enhanced_data.csv` | 2015-01 to 2025-04 | CSV | ~2,500 |

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `日期` / `date` | date | Trading date |
| `收盘价(元)` / `close_price` | float | Daily closing price (CNY) |
| `最高价(元)` / `high_price` | float | Daily high price (CNY) |
| `最低价(元)` / `low_price` | float | Daily low price (CNY) |
| `RSI1` | float | RSI (14-day), range [0, 100] |
| `K` | float | KDJ K-value, range [0, 100] |
| `CCI` | float | Commodity Channel Index, typical range [−200, 200] |
| `W&R` | float | Williams %R, range [0, 100] |
| `sentiment_score` | float | Composite sentiment indicator (computed if absent) |
| Other technical features | float | ~40 additional features |

### Sentiment Score Formula

If `sentiment_score` is not pre-computed, it is derived as:

```
sentiment_score = 0.35 × RSI + 0.35 × K + 0.20 × ((CCI + 200) / 4) + 0.10 × (100 − W%R)
```

### Target Variable

Generated at runtime:

```
target = 1  if  sentiment_score(t+1) − sentiment_score(t) > 10
         0  otherwise
```

This represents a "sentiment breakthrough" — a significant next-day increase in market sentiment.


## How to Obtain the Data

1. **Tushare (Free tier available):** Register at [tushare.pro](https://tushare.pro/), obtain an API token, and use `pro.fut_daily()` to pull CSI 300 futures data. Technical indicators can be computed with the `ta` or `TA-Lib` Python libraries.

2. **Wind Terminal:** If you have institutional access, use the Wind Excel/Python plugin to export daily OHLC + technical indicators for the IF continuous contract.

3. **CSMAR:** Available through most Chinese university library subscriptions. Export from the Derivatives database module.

After obtaining the data, rename columns to match the schema above and place files in the corresponding subdirectories.
