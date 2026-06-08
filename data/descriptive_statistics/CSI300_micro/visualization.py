import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置Matplotlib以支持中文显示和负号正常显示 ---
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]  # 使用黑体，根据系统安装的字体选择，如"WenQuanYi Micro Hei"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --- 加载数据 ---
# 假设文件在/home/ubuntu/upload/目录下
df = pd.read_csv("micro_data_cleaned.csv", encoding="utf-8") # 请确保此CSV文件与Python脚本在同一目录下，或者修改为您的实际文件路径

# --- 数据预处理 ---
# 将日期列转换为datetime对象
df["date"] = pd.to_datetime(df["date"])
# 将code和name列转换为分类类型，如果它们是固定的少数几个值
df["code"] = df["code"].astype("category")
df["name"] = df["name"].astype("category")

# --- 可视化函数定义 ---

def plot_time_series(dataframe, columns, title_prefix="", filename_prefix=""):
    """绘制时间序列折线图"""
    plt.figure(figsize=(15, 6))
    for col in columns:
        plt.plot(dataframe["date"], dataframe[col], label=col)
    plt.title(f"{title_prefix}时间序列趋势图")
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_time_series.png")
    plt.close()

def plot_distribution(dataframe, columns, plot_type="hist", filename_prefix=""):
    """绘制数值型数据的分布图 (直方图或KDE图)"""
    num_cols = len(columns)
    if num_cols == 0: return
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
    if num_cols == 1: axes = [axes] # Ensure axes is iterable for single plot

    for i, col in enumerate(columns):
        if plot_type == "hist":
            sns.histplot(dataframe[col], kde=True, ax=axes[i])
            axes[i].set_title(f"{col} - 直方图")
        elif plot_type == "kde":
            sns.kdeplot(dataframe[col], fill=True, ax=axes[i])
            axes[i].set_title(f"{col} - 核密度估计图")
        axes[i].set_xlabel("值")
        axes[i].set_ylabel("密度" if plot_type == "kde" else "频数")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{plot_type}_distribution.png")
    plt.close()

def plot_boxplots(dataframe, columns, filename_prefix=""):
    """绘制数值型数据的箱线图"""
    num_cols = len(columns)
    if num_cols == 0: return
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
    if num_cols == 1: axes = [axes]

    for i, col in enumerate(columns):
        sns.boxplot(y=dataframe[col], ax=axes[i])
        axes[i].set_title(f"{col} - 箱线图")
        axes[i].set_ylabel("值")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_boxplots.png")
    plt.close()

def plot_categorical_counts(dataframe, columns, filename_prefix=""):
    """绘制类别型数据的计数图"""
    num_cols = len(columns)
    if num_cols == 0: return
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))
    if num_cols == 1: axes = [axes]

    for i, col in enumerate(columns):
        sns.countplot(x=dataframe[col], ax=axes[i])
        axes[i].set_title(f"{col} - 计数图")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("计数")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_categorical_counts.png")
    plt.close()

def plot_correlation_heatmap(dataframe, numerical_cols, filename_prefix=""):
    """绘制数值型数据相关性热力图"""
    plt.figure(figsize=(18, 16))
    corr_matrix = dataframe[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("数值型指标相关性热力图")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_correlation_heatmap.png")
    plt.close()

# --- 定义不同类型的列 ---
# 时间序列相关指标 (价格、交易量等)
time_series_cols = [
    "open", "high", "low", "close", "settle_price",
    "volume", "turnover_million", "pct_change", "open_interest"
]

# 其他连续数值型指标 (技术指标)
other_numerical_cols = [
    "bias1", "bias2", "bias3", "boll", "boll_upper", "boll_lower",
    "cci", "kdj_k", "kdj_d", "kdj_j", "macd_dif", "macd_dea", "macd_hist",
    "rsi1", "rsi2", "rsi3", "ma5", "ma10", "ma20", "wr", "mtm", "mtmma",
    "obv", "po_net_holding_10k", "vr", "sd", "vs_basis", "turnover_rate_hs300"
]

# 类别型指标
categorical_cols = ["code", "name"]

# --- 生成图表 ---
print("开始生成时间序列图...")
plot_time_series(df, time_series_cols, title_prefix="CSI300股指期货", filename_prefix="csi300")
print("时间序列图生成完毕。")

print("开始生成数值型数据直方图...")
plot_distribution(df, other_numerical_cols, plot_type="hist", filename_prefix="csi300")
print("数值型数据直方图生成完毕。")

print("开始生成数值型数据KDE图...")
plot_distribution(df, other_numerical_cols, plot_type="kde", filename_prefix="csi300")
print("数值型数据KDE图生成完毕。")

print("开始生成数值型数据箱线图...")
plot_boxplots(df, other_numerical_cols, filename_prefix="csi300")
print("数值型数据箱线图生成完毕。")

print("开始生成类别型数据计数图...")
plot_categorical_counts(df, categorical_cols, filename_prefix="csi300")
print("类别型数据计数图生成完毕。")

# 所有数值型列，用于相关性分析
all_numerical_cols = time_series_cols + other_numerical_cols
print("开始生成相关性热力图...")
plot_correlation_heatmap(df, all_numerical_cols, filename_prefix="csi300")
print("相关性热力图生成完毕。")

print("所有图表生成完毕，请检查当前目录下的PNG文件。")

