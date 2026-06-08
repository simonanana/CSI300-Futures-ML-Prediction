import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# --- Configure Matplotlib for English display ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False # Ensure minus signs are displayed correctly

# --- Load Data ---
data_file_path = "宏观数据10-15.xlsx"
# Fallback to absolute path if running in sandbox environment
if not os.path.exists(data_file_path):
    data_file_path = "/Downloads/宏观数据10-15.xlsx"

# Load the Excel file, specifying the sheet name and skipping initial metadata rows
try:
    df = pd.read_excel(data_file_path, sheet_name="中国_CPI_定基指数", header=1)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Drop the first 4 rows which contain metadata like '频率', '单位', '指标ID', '时间区间'
df = df.iloc[4:].copy()

# Rename the first column to 'Date' for clarity
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where 'Date' is NaT (Not a Time) after conversion
df.dropna(subset=['Date'], inplace=True)

# Set 'Date' as index
df.set_index('Date', inplace=True)

# Convert all other columns to numeric, coercing errors will turn non-numeric to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop columns that are entirely NaN after numeric conversion (e.g., empty columns or columns with only text metadata)
df.dropna(axis=1, how='all', inplace=True)

# --- Rename Chinese Columns to English for better readability in plots ---
column_mapping = {
    "中国:CPI:定基指数": "China: CPI: Fixed Base Index",
    "中国:PPI:定基指数": "China: PPI: Fixed Base Index",
    "中国:PPIRM:累计同比": "China: PPIRM: YTD YoY",
    "上证综合指数": "Shanghai Composite Index",
    "上证综合指数:涨跌幅": "Shanghai Composite Index: Change Rate",
    "陆股通:当日成交金额(人民币)": "Northbound Capital: Daily Turnover (RMB)",
    "沪深300指数": "CSI300 Index",
    "创业板指数": "ChiNext Index",
    "创业板指数:涨跌幅": "ChiNext Index: Change Rate",
    "中国:国债到期收益率:10年": "China: 10Y Treasury Yield",
    "即期汇率:美元兑离岸人民币(USDCNH)": "Spot Exchange Rate: USD to CNH",
    "美国:国债收益率:10年": "US: 10Y Treasury Yield",
    "美国:有效联邦基金利率(EFFR)": "US: Effective Federal Funds Rate (EFFR)",
    "中国:公开市场操作:货币净投放": "China: Open Market Operations: Net Monetary Injection"
}
df.rename(columns=column_mapping, inplace=True)

# --- Handle Missing Values (Interpolation for time series data) ---
df_filled = df.interpolate(method='time', limit_direction='both')

# --- Visualization Function Definitions ---

def plot_time_series(dataframe, columns, title_suffix="", filename_prefix="", cols_per_row=2):
    """Plot time series line charts for selected columns, with dual axis for specific pairs"""
    num_plots = len(columns)
    if num_plots == 0: return

    # Define specific dual-axis pairs
    dual_axis_pairs = [
        ("Shanghai Composite Index", "Shanghai Composite Index: Change Rate"),
        ("China: CPI: Fixed Base Index", "China: 10Y Treasury Yield"),
        ("CSI300 Index", "ChiNext Index: Change Rate")
    ]

    plotted_cols = set()
    for col1, col2 in dual_axis_pairs:
        if col1 in columns and col2 in columns and col1 not in plotted_cols and col2 not in plotted_cols:
            plt.figure(figsize=(15, 6))
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(dataframe.index, dataframe[col1], label=col1, color='blue')
            ax2.plot(dataframe.index, dataframe[col2], label=col2, color='red', linestyle='--')

            ax1.set_xlabel("Date")
            ax1.set_ylabel(col1, color='blue')
            ax2.set_ylabel(col2, color='red')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.title(f"Time Series: {col1} vs {col2}")
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_{col1.replace(':', '_').replace(' ', '_')}_vs_{col2.replace(':', '_').replace(' ', '_')}_timeseries.png")
            plt.close()
            plotted_cols.add(col1)
            plotted_cols.add(col2)

    # Plot remaining columns as single time series plots
    remaining_cols = [col for col in columns if col not in plotted_cols]
    if not remaining_cols: return

    rows = math.ceil(len(remaining_cols) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(8 * cols_per_row, 5 * rows))
    axes = axes.flatten() if rows > 1 or cols_per_row > 1 else [axes]

    for i, col in enumerate(remaining_cols):
        ax = axes[i]
        ax.plot(dataframe.index, dataframe[col], label=col)
        ax.set_title(f"Time Series: {col}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    for j in range(len(remaining_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_remaining_timeseries.png")
    plt.close()

def plot_grid_distributions(dataframe, columns, plot_type="hist", cols_per_row=4, filename_prefix=""):
    """Generic function to plot distributions (hist/kde/box) in a grid layout"""
    num_plots = len(columns)
    if num_plots == 0: return

    rows = math.ceil(num_plots / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 5 * rows))
    axes = axes.flatten() if rows > 1 or cols_per_row > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        if plot_type == "hist":
            sns.histplot(dataframe[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} - Histogram")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        elif plot_type == "kde":
            sns.kdeplot(dataframe[col].dropna(), fill=True, ax=ax)
            ax.set_title(f"{col} - KDE Plot")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
        elif plot_type == "boxplot":
            sns.boxplot(y=dataframe[col].dropna(), ax=ax)
            ax.set_title(f"{col} - Box Plot")
            ax.set_ylabel("Value")

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{plot_type}_plots.png")
    plt.close()

def plot_correlation_heatmap(dataframe, numerical_cols, filename_prefix=""):
    """Plot correlation heatmap for numerical data"""
    plt.figure(figsize=(18, 16))
    corr_matrix = dataframe[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Heatmap of Macro Indicators")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_correlation_heatmap.png")
    plt.close()

# --- Define column groups for visualization ---
# All numerical columns after renaming
numerical_cols = df_filled.columns.tolist()

# --- Generate Charts ---
print("Generating time series plots...")
plot_time_series(df_filled, numerical_cols, title_suffix="Macro Data", filename_prefix="macro")
print("Time series plots generated.")

print("Generating numerical data histograms...")
plot_grid_distributions(df_filled, numerical_cols, plot_type="hist", cols_per_row=4, filename_prefix="macro")
print("Numerical data histograms generated.")

print("Generating numerical data KDE plots...")
plot_grid_distributions(df_filled, numerical_cols, plot_type="kde", cols_per_row=4, filename_prefix="macro")
print("Numerical data KDE plots generated.")

print("Generating numerical data box plots...")
plot_grid_distributions(df_filled, numerical_cols, plot_type="boxplot", cols_per_row=4, filename_prefix="macro")
print("Numerical data box plots generated.")

print("Generating correlation heatmap...")
plot_correlation_heatmap(df_filled, numerical_cols, filename_prefix="macro")
print("Correlation heatmap generated.")

print("All macro data charts generated. Please check the PNG files in the current directory.")

