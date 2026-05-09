# ===============================================================
# YOLOv7 训练与验证独立模块（兼容 YOLOv8 结构）
# ===============================================================

import os
import subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from glob import glob


# ===============================================================
# 🔍 递归扫描图片路径
# ===============================================================
def collect_images_recursive(root_dir, output_txt):
    """递归收集指定目录下的所有图片路径并写入 txt 文件"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    paths = [p.replace("\\", "/") for p in paths]
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    return len(paths)


# ===============================================================
# 🚀 YOLOv7 训练 + 验证 + 可视化
# ===============================================================
def train_yolov7(model_name, project_root):
    """
    YOLOv7 官方检测版训练流程（统一保存到 runs_seg/<model_name>/ 下）
    输出结构：
        runs_seg/<model_name>/
            ├── weights/best.pt
            ├── evaluation_metrics.csv
            ├── metrics_bar.png
            ├── metrics_radar.png
            └── <model_name>_val/results.txt
    """
    yolov7_root = os.path.join(project_root, "yolov7")
    runs_dir = os.path.join(project_root, "runs_seg")
    exp_dir = os.path.join(runs_dir, model_name)
    os.makedirs(exp_dir, exist_ok=True)

    weights_path = os.path.join(yolov7_root, "yolov7.pt")
    cfg_path = os.path.join(yolov7_root, "cfg", "training", "yolov7.yaml")
    data_path = os.path.join(project_root, "data.yaml")
    fixed_data_path = os.path.join(project_root, "data_yolov7.yaml")
    best_path = os.path.join(exp_dir, "weights", "best.pt")

    # ======= 检查是否已训练过 =======
    skip_train = os.path.exists(best_path)
    if skip_train:
        print(f"⏩ 检测到 {model_name} 已存在 best.pt，将跳过训练直接验证。")

    # ======= 生成 YOLOv7 专用 data.yaml =======
    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = project_root.replace("\\", "/")
    train_dir = os.path.join(base_dir, data_cfg.get("train", "train"))
    val_dir = os.path.join(base_dir, data_cfg.get("val", "val"))
    test_dir = os.path.join(base_dir, data_cfg.get("test", "test"))

    train_txt = os.path.join(project_root, "train_list.txt")
    val_txt = os.path.join(project_root, "val_list.txt")
    test_txt = os.path.join(project_root, "test_list.txt")
    collect_images_recursive(train_dir, train_txt)
    collect_images_recursive(val_dir, val_txt)
    collect_images_recursive(test_dir, test_txt)

    fixed_yaml = {
        "train": train_txt,
        "val": val_txt,
        "test": test_txt,
        "nc": data_cfg.get("nc", 2),
        "names": data_cfg.get("names", {0: "class0", 1: "class1"}),
    }
    with open(fixed_data_path, "w", encoding="utf-8") as f:
        yaml.dump(fixed_yaml, f, sort_keys=False, allow_unicode=True)

    # ======= 训练阶段 =======
    if not skip_train:
        print(f"\n==== 开始训练模型: {model_name} (YOLOv7 Detection) ====\n")
        train_cmd = [
            "python", "train.py",
            "--weights", weights_path,
            "--cfg", cfg_path,
            "--data", fixed_data_path,
            "--epochs", "100",
            "--batch-size", "8",
            "--img", "512",
            "--project", runs_dir,
            "--name", model_name,
            "--device", "0"
        ]
        subprocess.run(train_cmd, cwd=yolov7_root, check=True)

        if not os.path.exists(best_path):
            raise FileNotFoundError(f"❌ 未找到 {best_path}，请检查训练是否成功。")

    # ======= 验证阶段 =======
    print(f"🔍 开始验证 {model_name} 模型性能 ...")
    val_name = f"{model_name}_val"
    val_cmd = [
        "python", "test.py",
        "--weights", best_path,
        "--data", fixed_data_path,
        "--batch-size", "8",
        "--img", "512",
        "--task", "test",
        "--project", runs_dir,
        "--name", val_name,
        "--device", "0",
    ]

    result = subprocess.run(val_cmd, cwd=yolov7_root, capture_output=True, text=True)

    # ======= 保存验证日志 =======
    val_log_path = os.path.join(runs_dir, val_name, "results.txt")
    os.makedirs(os.path.dirname(val_log_path), exist_ok=True)
    with open(val_log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"✅ 已保存验证日志到: {val_log_path}")

    # ======= 解析验证结果 =======
    metrics = {}
    with open(val_log_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        for line in lines:
            parts = line.split()
            # 兼容 all / overall / mean 行
            if any(x.lower() in ("all", "overall", "mean") for x in parts[:2]) and len(parts) >= 7:
                try:
                    metrics = {
                        "metrics/precision(B)": float(parts[-4]),
                        "metrics/recall(B)": float(parts[-3]),
                        "metrics/mAP50(B)": float(parts[-2]),
                        "metrics/mAP50-95(B)": float(parts[-1]),
                    }
                    break
                except ValueError:
                    continue

    # ======= fallback：尝试读取默认 YOLOv7 results.txt =======
    if not metrics:
        default_results = os.path.join(yolov7_root, "runs", "test", "exp", "results.txt")
        if os.path.exists(default_results):
            with open(default_results, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                last_line = lines[-1].split()
                if len(last_line) >= 4:
                    metrics = {
                        "metrics/precision(B)": float(last_line[0]),
                        "metrics/recall(B)": float(last_line[1]),
                        "metrics/mAP50(B)": float(last_line[2]),
                        "metrics/mAP50-95(B)": float(last_line[3]),
                    }

    if metrics:
        print(f"📊 成功解析 YOLOv7 验证结果: {metrics}")
    else:
        print("⚠️ 未在日志或结果文件中找到指标信息。")

    # ======= 保存指标与图表 =======
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    metrics_path = os.path.join(exp_dir, "evaluation_metrics.csv")

    for _ in range(3):
        try:
            df.to_csv(metrics_path, index=False)
            break
        except PermissionError:
            print(f"⚠️ 文件被占用，2 秒后重试写入: {metrics_path}")
            time.sleep(2)
    else:
        print(f"❌ 无法写入 {metrics_path}，请关闭该文件后重试。")

    if not df.empty:
        # 柱状图
        plt.figure(figsize=(7, 5))
        plt.bar(df["Metric"], df["Value"], color="orange", alpha=0.8)
        plt.title(f"Performance Metrics - {model_name}")
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "metrics_bar.png"))
        plt.close()

        # 雷达图
        labels = df["Metric"]
        values = df["Value"].tolist()
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color="darkorange", linewidth=2)
        ax.fill(angles, values, color="gold", alpha=0.4)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title(f"Radar Metrics - {model_name}")
        plt.savefig(os.path.join(exp_dir, "metrics_radar.png"))
        plt.close()

    print(f"✅ YOLOv7 模型 {model_name} 的指标与图表已保存到：{exp_dir}")
    return metrics, exp_dir


# ===============================================================
# 支持单独运行
# ===============================================================
if __name__ == "__main__":
    project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    train_yolov7("yolov7", project_root)
