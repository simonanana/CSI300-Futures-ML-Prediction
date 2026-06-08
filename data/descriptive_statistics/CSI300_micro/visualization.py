"""
CSI 300 Index Futures — Exploratory Data Visualization
=======================================================
Project : CSI 300 Futures ML Prediction
         (Direction · Volatility · Sentiment)
Author  : NTU MH6822 Research Group
Dataset : micro_data_cleaned.csv

Output files
------------
  csi300_time_series.png
  csi300_hist_plots.png
  csi300_kde_plots.png
  csi300_boxplot_plots.png
  csi300_correlation_heatmap.png
  numerical_boxplot_plots.png
  numerical_density_plots.png
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

PALETTE_LINE = sns.color_palette("tab10", 10)
PALETTE_DIST = sns.color_palette("husl", 8)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("micro_data_cleaned.csv", encoding="utf-8")

# ── Preprocessing ─────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
df["code"] = df["code"].astype("category")
df["name"] = df["name"].astype("category")

# ── Column groups ─────────────────────────────────────────────────────────────
PRICE_COLS = ["open", "high", "low", "close", "settle_price"]
VOLUME_COLS = ["volume", "turnover_million", "open_interest"]
RETURN_COLS = ["pct_change"]

TIME_SERIES_COLS = PRICE_COLS + VOLUME_COLS + RETURN_COLS

TECH_COLS = [
    "bias1", "bias2", "bias3",
    "boll", "boll_upper", "boll_lower",
    "cci",
    "kdj_k", "kdj_d", "kdj_j",
    "macd_dif", "macd_dea", "macd_hist",
    "rsi1", "rsi2", "rsi3",
    "ma5", "ma10", "ma20",
    "wr", "mtm", "mtmma",
    "obv", "po_net_holding_10k", "vr", "sd", "vs_basis",
    "turnover_rate_hs300",
]

ALL_NUMERICAL_COLS = TIME_SERIES_COLS + TECH_COLS


# ── Helper: print descriptive stats to stdout ─────────────────────────────────
def print_descriptive_stats(dataframe: pd.DataFrame, cols: list, label: str) -> None:
    """Print a tidy descriptive statistics table for the given columns."""
    stats = dataframe[cols].describe().T
    stats["skewness"] = dataframe[cols].skew()
    stats["kurtosis"] = dataframe[cols].kurt()
    print(f"\n{'=' * 60}")
    print(f"  Descriptive Statistics — {label}")
    print(f"{'=' * 60}")
    print(stats.to_string())


# ── Plot 1 : Time-series overview (csi300_time_series.png) ────────────────────
def plot_time_series(dataframe: pd.DataFrame) -> None:
    """
    Multi-panel time-series plot.
    Panel 1 — OHLC prices + settle price
    Panel 2 — Volume, open interest, turnover
    Panel 3 — Daily percentage change
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("CSI 300 Index Futures — Time-Series Overview", fontsize=15, fontweight="bold", y=1.01)

    # Panel 1 — prices
    for col, color in zip(PRICE_COLS, PALETTE_LINE):
        axes[0].plot(dataframe["date"], dataframe[col], label=col.replace("_", " ").title(), color=color, linewidth=0.8)
    axes[0].set_ylabel("Price (CNY)")
    axes[0].set_title("OHLC & Settle Price")
    axes[0].legend(ncol=5, fontsize=8, loc="upper left")

    # Panel 2 — volume & OI (dual y-axis)
    ax2_r = axes[1].twinx()
    axes[1].bar(dataframe["date"], dataframe["volume"], label="Volume", color="#4e79a7", alpha=0.5, width=1.5)
    ax2_r.plot(dataframe["date"], dataframe["open_interest"], label="Open Interest", color="#f28e2b", linewidth=0.9)
    axes[1].set_ylabel("Volume (contracts)")
    ax2_r.set_ylabel("Open Interest")
    axes[1].set_title("Volume & Open Interest")
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2_r.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="upper left")

    # Panel 3 — returns
    positive = dataframe["pct_change"] >= 0
    axes[2].bar(dataframe.loc[positive, "date"],  dataframe.loc[positive, "pct_change"],  color="#2ca02c", alpha=0.7, width=1.5, label="Positive")
    axes[2].bar(dataframe.loc[~positive, "date"], dataframe.loc[~positive, "pct_change"], color="#d62728", alpha=0.7, width=1.5, label="Negative")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylabel("Daily Return (%)")
    axes[2].set_title("Daily Percentage Change")
    axes[2].set_xlabel("Date")
    axes[2].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("csi300_time_series.png")
    plt.close()
    print("  [✓] csi300_time_series.png")


# ── Plot 2 : Histograms — CSI300 price/volume cols (csi300_hist_plots.png) ────
def plot_hist(dataframe: pd.DataFrame, cols: list, filename: str, title: str) -> None:
    """Histogram with KDE overlay for each column."""
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        sns.histplot(dataframe[col].dropna(), kde=True, ax=axes[i], color=PALETTE_DIST[i % len(PALETTE_DIST)])
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  [✓] {filename}")


# ── Plot 3 : KDE plots (csi300_kde_plots.png) ─────────────────────────────────
def plot_kde(dataframe: pd.DataFrame, cols: list, filename: str, title: str) -> None:
    """Kernel Density Estimate plots with filled area."""
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        sns.kdeplot(dataframe[col].dropna(), fill=True, ax=axes[i], color=PALETTE_DIST[i % len(PALETTE_DIST)])
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  [✓] {filename}")


# ── Plot 4 : Boxplots — CSI300 cols (csi300_boxplot_plots.png) ────────────────
def plot_boxplots(dataframe: pd.DataFrame, cols: list, filename: str, title: str) -> None:
    """Boxplots with individual data points for outlier inspection."""
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        sns.boxplot(y=dataframe[col].dropna(), ax=axes[i], color=PALETTE_DIST[i % len(PALETTE_DIST)],
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_ylabel("Value")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  [✓] {filename}")


# ── Plot 5 : Correlation heatmap (csi300_correlation_heatmap.png) ─────────────
def plot_correlation_heatmap(dataframe: pd.DataFrame, cols: list) -> None:
    """
    Pearson correlation heatmap for all numerical features.
    Only the lower triangle is annotated to reduce clutter.
    """
    corr = dataframe[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # mask upper triangle

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Heatmap — All Numerical Variables", fontsize=15, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig("csi300_correlation_heatmap.png")
    plt.close()
    print("  [✓] csi300_correlation_heatmap.png")


# ── Plot 6 : Technical indicator boxplots (numerical_boxplot_plots.png) ────────
def plot_numerical_boxplots(dataframe: pd.DataFrame, cols: list) -> None:
    """
    Side-by-side boxplots for all technical indicator columns,
    with z-score normalisation to enable visual comparison on one axis.
    """
    df_norm = dataframe[cols].apply(lambda x: (x - x.mean()) / x.std())
    df_melt = df_norm.melt(var_name="Feature", value_name="Z-score")

    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(data=df_melt, x="Feature", y="Z-score", palette="Set3", ax=ax,
                flierprops=dict(marker="o", markersize=1.5, alpha=0.3))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Technical Indicators — Normalised Boxplots (Z-score)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=60, labelsize=8)

    plt.tight_layout()
    plt.savefig("numerical_boxplot_plots.png")
    plt.close()
    print("  [✓] numerical_boxplot_plots.png")


# ── Plot 7 : Technical indicator density (numerical_density_plots.png) ─────────
def plot_numerical_density(dataframe: pd.DataFrame, cols: list) -> None:
    """
    Overlaid KDE curves for all technical indicators on a single axes.
    Useful for comparing distributional shapes across features.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, col in enumerate(cols):
        series = dataframe[col].dropna()
        z = (series - series.mean()) / series.std()
        sns.kdeplot(z, ax=ax, label=col, linewidth=0.9, alpha=0.75,
                    color=sns.color_palette("tab20", len(cols))[i])

    ax.set_title("Technical Indicators — Density Comparison (Z-score Normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.legend(ncol=4, fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig("numerical_density_plots.png")
    plt.close()
    print("  [✓] numerical_density_plots.png")


# ── Main execution ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nCSI 300 Index Futures — Generating Visualizations")
    print("=" * 55)

    # Descriptive statistics (console output)
    print_descriptive_stats(df, PRICE_COLS + VOLUME_COLS + RETURN_COLS, "Price & Volume Features")
    print_descriptive_stats(df, TECH_COLS, "Technical Indicators")

    # Charts
    print("\nGenerating plots…")
    plot_time_series(df)

    plot_hist(df, TIME_SERIES_COLS,
              "csi300_hist_plots.png",
              "CSI 300 Futures — Price & Volume Distributions (Histogram)")

    plot_kde(df, TIME_SERIES_COLS,
             "csi300_kde_plots.png",
             "CSI 300 Futures — Price & Volume Distributions (KDE)")

    plot_boxplots(df, TIME_SERIES_COLS,
                  "csi300_boxplot_plots.png",
                  "CSI 300 Futures — Price & Volume Boxplots")

    plot_correlation_heatmap(df, ALL_NUMERICAL_COLS)

    plot_numerical_boxplots(df, TECH_COLS)
    plot_numerical_density(df, TECH_COLS)

    print("\nAll plots saved successfully. Check the current directory for PNG files.\n")
