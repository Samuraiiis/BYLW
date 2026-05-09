# ===============================================================
# 功能：
#  - YOLOv8n / YOLOv8s / YOLOv8n+CBAM+BRH / YOLOv7-seg 自动训练与验证
#  - 自动保存日志、指标表、柱状图与雷达图
#  - 自动生成综合对比表（CSV + 图像）
#  - 自动在 test 集上评估最优模型性能
# 环境：Win11 + YOLOv8 v8.3.221 + YOLOv7 clone + RTX4060(8GB)
# ===============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from train_yolov7_runner import train_yolov7
import shutil
import numpy as np
from ultralytics import YOLO
from cbam_brh import inject_custom_modules
import matplotlib

# ✅ 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ===============================================================
# 🚀 YOLOv8 单模型训练 + 验证 + 可视化
def train_and_evaluate(model_name, project_root, inject_modules=False, skip_train=False):
    """YOLOv8 模型训练 + 验证 + 可视化（全部结果保存在 runs_seg/ 下）"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    runs_dir = os.path.join(project_root, "runs_seg")
    train_name = model_name + ("_CBAM_BRH" if inject_modules else "")
    exp_dir = os.path.join(runs_dir, train_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n==== {'验证已有模型' if skip_train else '开始训练模型'}: {train_name} ====")

    model = YOLO(f"{model_name}.pt")
    if inject_modules:
        model = inject_custom_modules(model)

    best_path = os.path.join(exp_dir, "weights", "best.pt")

    # ======= 训练阶段 =======
    if not skip_train:
        model.train(
            data="data.yaml",
            imgsz=512,
            epochs=100,
            batch=10,
            lr0=0.0015,
            optimizer="AdamW",
            weight_decay=0.0005,
            cos_lr=True,
            mosaic=0.2,
            hsv_v=0.5,
            degrees=5,
            translate=0.05,
            scale=0.3,
            flipud=0.2,
            fliplr=0.5,
            cutmix=0.1,
            mixup=0.1,
            amp=True,
            patience=15,
            project=runs_dir,
            name=train_name
        )
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"❌ 未找到 {best_path}，请检查训练是否成功。")
    else:
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"⚠️ 跳过训练但未找到 {best_path}，请确认路径。")

    # ======= 验证阶段 =======
    print(f"🔍 开始验证 {train_name} 模型性能 ...")
    val_model = YOLO(best_path)
    val_results = val_model.val(split="val", project=runs_dir, name=f"{train_name}_val")
    metrics = val_results.results_dict

    print("\n📊 模型性能指标：")
    for key, val in metrics.items():
        print(f"{key:<20}{val:.4f}")

    # ======= 保存指标与图表 =======
    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    df_metrics["Metric"] = df_metrics["Metric"].str.replace("metrics/", "", regex=False)
    df_metrics.to_csv(os.path.join(exp_dir, "evaluation_metrics.csv"), index=False)

    # 柱状图
    plt.figure(figsize=(7, 5))
    plt.bar(df_metrics["Metric"], df_metrics["Value"], color="steelblue", alpha=0.8)
    plt.title(f"Performance Metrics - {train_name}")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "metrics_bar.png"))
    plt.close()

    # 雷达图
    labels = df_metrics["Metric"]
    values = df_metrics["Value"].tolist()
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="teal", linewidth=2)
    ax.fill(angles, values, color="lightblue", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"Radar Metrics - {train_name}")
    plt.savefig(os.path.join(exp_dir, "metrics_radar.png"))
    plt.close()

    print(f"✅ 模型 {train_name} 的日志与图表已保存到：{exp_dir}")
    return metrics, exp_dir


# ===============================================================
# 📊 对比分析 + 图表导出
def export_comparison(all_results, export_dir):
    os.makedirs(export_dir, exist_ok=True)

    df = pd.DataFrame(all_results).T  # index 为模型名
    df.index.name = "Model"

    # 🧩 自动补齐缺失前缀
    df.columns = [
        c if str(c).startswith("metrics/") else f"metrics/{c}"
        for c in df.columns
    ]

    # 🩹 防空检查
    if df.empty or df.isna().all().all():
        print("⚠️ 未检测到有效指标数据，跳过图表生成。")
        return

    df = df.apply(pd.to_numeric, errors="ignore")
    shared_cols = [c for c in df.columns if not df[c].isna().all()]
    df = df[shared_cols]

    csv_path = os.path.join(export_dir, "comparison.csv")
    df.to_csv(csv_path, index=True, float_format="%.4f")
    print(f"✅ 对比结果已导出: {csv_path}")

    # ===== 绘制对比图 =====
    metrics_to_plot = []
    for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
        if k in df.columns:
            metrics_to_plot.append(k)
    for k in ["metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
        if k in df.columns:
            metrics_to_plot.append(k)

    if len(metrics_to_plot) == 0:
        print("⚠️ 没有检测到可绘制的指标，跳过图表生成。")
        return

    plt.figure(figsize=(10, 6))
    df_plot = df[metrics_to_plot]
    df_plot.plot(kind="bar")
    plt.title("YOLO 模型性能对比", fontsize=14)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(export_dir, "comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"📊 可视化图表已保存: {fig_path}")



# ===============================================================
# 🚀 主函数
def main():
    project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    runs_dir = os.path.join(project_root, "runs_seg")

    models_to_run = [
        ("yolov8n-seg", False),
        ("yolov8s-seg", False),
        ("yolov8n-seg", True),
        ("yolov7", False)
    ]

    all_results = {}

    for model_name, inject_flag in models_to_run:
        train_name = model_name + ("_CBAM_BRH" if inject_flag else "")
        yolov8_best = os.path.join(runs_dir, train_name, "weights", "best.pt")
        yolov7_best = os.path.join(runs_dir, "yolov7", "weights", "best.pt")

        if "yolov7" in model_name:
            if os.path.exists(yolov7_best):
                print(f"⏩ 检测到 {train_name} 已存在 best.pt，跳过训练但执行验证。")
            metrics, exp_path = train_yolov7(model_name, project_root)
        else:
            if os.path.exists(yolov8_best):
                print(f"⏩ 检测到 {train_name} 已存在 best.pt，跳过训练但执行验证。")
                metrics, exp_path = train_and_evaluate(model_name, project_root,
                                                       inject_modules=inject_flag, skip_train=True)
            else:
                metrics, exp_path = train_and_evaluate(model_name, project_root,
                                                       inject_modules=inject_flag, skip_train=False)

        all_results[train_name] = metrics

    export_dir = os.path.join(project_root, "train_results", "comparison")
    export_comparison(all_results, export_dir)


# ===============================================================
if __name__ == "__main__":
    main()
