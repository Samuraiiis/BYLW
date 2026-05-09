import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================== 中文显示设置 ==================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 基本配置 ==================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
runs_dir = os.path.join(project_root, "runs_seg")
export_dir = os.path.join(project_root, "train_results", "comparison_fixed")
os.makedirs(export_dir, exist_ok=True)

# ================== 模型列表 ==================
models = [
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8n-seg_CBAM_BRH",
    "yolov7",
]


# ================== 字段标准化函数 ==================
def normalize_metric_name(name: str) -> str:
    """统一不同来源模型的指标命名格式"""
    name = name.strip().lower()
    name = name.replace("@0.5:0.95", "50-95").replace("@0.5", "50")
    name = name.replace("map", "mAP")
    name = name.replace(" ", "").replace(":", "").replace("_", "")
    # 统一 key 格式
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50-95": "metrics/mAP50-95(B)",
    }
    for k, v in mapping.items():
        if k in name:
            return v
    if not name.startswith("metrics/"):
        name = "metrics/" + name
    return name


# ================== 汇总结果 ==================
all_metrics = {}

for model in models:
    csv_path = os.path.join(runs_dir, model, "evaluation_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ 未找到指标文件: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if "Metric" not in df.columns or "Value" not in df.columns:
        print(f"⚠️ {model} 文件格式异常，跳过。")
        continue

    # 统一字段名
    df["Metric"] = df["Metric"].apply(normalize_metric_name)
    metrics_dict = dict(zip(df["Metric"], df["Value"]))
    all_metrics[model] = metrics_dict

# ================== 生成汇总表 ==================
if not all_metrics:
    raise RuntimeError("❌ 未找到任何指标文件。")

df_all = pd.DataFrame(all_metrics).T
df_all = df_all.apply(pd.to_numeric, errors="ignore")

# 自动排序（按 mAP50(B)）
if "metrics/mAP50(B)" in df_all.columns:
    df_all = df_all.sort_values(by="metrics/mAP50(B)", ascending=False)

# 导出
csv_path = os.path.join(export_dir, "final_metrics_comparison_fixed.csv")
df_all.to_csv(csv_path, float_format="%.4f")
print(f"✅ 修正版指标对比表已导出：{csv_path}")

# ================== 绘制柱状图 ==================
metrics_to_plot = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
metrics_to_plot = [m for m in metrics_to_plot if m in df_all.columns]

plt.figure(figsize=(10, 6))
df_all[metrics_to_plot].plot(kind="bar")
plt.title("YOLO 模型性能对比（修正版）", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "final_metrics_bar_fixed.png"), dpi=300)
plt.close()
print("📊 柱状图已生成。")

# ================== 绘制雷达图 ==================
def plot_radar(df, export_path):
    labels = metrics_to_plot
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for idx, row in df.iterrows():
        values = [row[m] for m in labels]
        values += values[:1]
        ax.plot(angles, values, label=idx, linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("YOLO 模型指标雷达图", fontsize=13)
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05))
    plt.tight_layout()
    plt.savefig(export_path, dpi=300)
    plt.close()

plot_radar(df_all, os.path.join(export_dir, "final_metrics_radar_fixed.png"))
print("🧭 雷达图已生成。")

# ================== 输出综合结论 ==================
if "metrics/mAP50(B)" in df_all.columns:
    best_model = df_all["metrics/mAP50(B)"].idxmax()
    print(f"\n🏆 最优模型：{best_model}")
    print(df_all.loc[best_model])

print("\n✅ 全部完成！")
