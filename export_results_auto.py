"""
自动扫描 runs_seg 下所有模型的指标文件，生成综合性能对比报告
作者: 你自己 :)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def export_comparison_auto(runs_dir, export_dir=None):
    """
    🚀 自动扫描 runs_seg 下的所有模型目录，汇总 evaluation_metrics.csv 生成对比图表
    参数:
        runs_dir (str): runs_seg 目录路径，例如 E:/PyCharm.../Projects/EI_PROJ/runs_seg
        export_dir (str): 输出目录（默认 runs_dir/comparison）
    """
    print(f"\n🔍 正在扫描 {runs_dir} 下的模型指标文件 ...")
    if export_dir is None:
        export_dir = os.path.join(runs_dir, "comparison")
    os.makedirs(export_dir, exist_ok=True)

    # === 搜索所有 evaluation_metrics.csv ===
    metrics_files = []
    for root, dirs, files in os.walk(runs_dir):
        for f in files:
            if f == "evaluation_metrics.csv":
                metrics_files.append(os.path.join(root, f))
    if not metrics_files:
        print("⚠️ 未找到任何 evaluation_metrics.csv 文件。")
        return

    print(f"✅ 找到 {len(metrics_files)} 个模型指标文件。")

    # === 读取每个模型指标 ===
    model_results = {}
    for csv_path in metrics_files:
        model_name = os.path.basename(os.path.dirname(csv_path))
        try:
            df = pd.read_csv(csv_path)
            metric_dict = dict(zip(df["Metric"], df["Value"]))
            model_results[model_name] = metric_dict
        except Exception as e:
            print(f"⚠️ 无法读取 {csv_path}: {e}")

    # === 合并成 DataFrame 并导出 CSV ===
    df_all = pd.DataFrame(model_results).T
    df_all.index.name = "Model"
    df_all = df_all.apply(pd.to_numeric, errors="ignore")
    csv_out = os.path.join(export_dir, "comparison.csv")
    df_all.to_csv(csv_out, float_format="%.4f")
    print(f"📊 汇总表已导出: {csv_out}")

    # === 选择可视化指标 ===
    numeric_cols = [c for c in df_all.columns if df_all[c].dtype != "object" and not df_all[c].isna().all()]
    if not numeric_cols:
        print("⚠️ 没有数值型指标可绘图，跳过。")
        return

    # === 绘制柱状对比图 ===
    plt.figure(figsize=(10, 6))
    df_all[numeric_cols].plot(kind="bar")
    plt.title("YOLO 模型性能综合对比", fontsize=14)
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "comparison.png"), dpi=300)
    plt.close()
    print(f"✅ 对比图已保存: {os.path.join(export_dir, 'comparison.png')}")

    print("🎯 自动扫描汇总完成。")


if __name__ == "__main__":
    project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    runs_dir = os.path.join(project_root, "runs_seg")
    export_comparison_auto(runs_dir)
