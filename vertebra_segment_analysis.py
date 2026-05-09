import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# ================== 中文显示设置 ==================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 基本配置 ==================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
test_dir = os.path.join(project_root, "test")
runs_dir = os.path.join(project_root, "runs_seg")
export_dir = os.path.join(project_root, "vertebra_analysis")
os.makedirs(export_dir, exist_ok=True)

# ================== 模型列表 ==================
models = [
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8n-seg_CBAM_BRH",
    "yolov7",
]

# ================== 椎体节段定义 ==================
# 从测试数据目录结构中提取椎体节段信息
vertebra_segments = []
for view_dir in os.listdir(test_dir):
    view_path = os.path.join(test_dir, view_dir)
    if os.path.isdir(view_path):
        for seg_dir in os.listdir(view_path):
            if seg_dir not in vertebra_segments:
                vertebra_segments.append(seg_dir)

print(f"识别到的椎体节段: {vertebra_segments}")

# ================== 读取标签文件 ==================
def read_labels(label_path):
    """读取YOLO格式的标签文件"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append([class_id, x_center, y_center, width, height])
    return labels

# ================== 椎体节段识别 ==================
def identify_vertebra_segment(image_path, labels):
    """根据图像路径和标签识别椎体节段"""
    # 从图像路径中提取节段信息
    path_parts = image_path.split(os.sep)
    for part in path_parts:
        if '-' in part and all(char.isalpha() or char.isdigit() or char == '-' for char in part):
            # 检查是否是椎体节段格式，如 L3-L5
            if any(seg in part for seg in vertebra_segments):
                return part
    return "Unknown"

# ================== 分析单椎体识别结果 ==================
def analyze_vertebra_segmentation():
    """分析单椎体识别与定位结果"""
    results = []
    
    for model in models:
        model_dir = os.path.join(runs_dir, model)
        if not os.path.exists(model_dir):
            print(f"⚠️ 模型目录不存在: {model_dir}")
            continue
        
        # 查找模型的预测结果
        pred_dir = os.path.join(model_dir, "predictions")
        if not os.path.exists(pred_dir):
            # 尝试其他可能的预测结果目录
            pred_dir = os.path.join(model_dir, "test")
            if not os.path.exists(pred_dir):
                print(f"⚠️ 预测结果目录不存在: {model_dir}")
                continue
        
        # 获取所有预测图像
        pred_images = glob(os.path.join(pred_dir, "*.jpg")) + glob(os.path.join(pred_dir, "*.png"))
        
        for img_path in pred_images:
            # 提取原始图像路径
            img_name = os.path.basename(img_path)
            
            # 查找对应的标签文件
            label_path = os.path.join(os.path.dirname(img_path), img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            
            # 读取标签
            labels = read_labels(label_path)
            
            # 识别椎体节段
            segment = identify_vertebra_segment(img_path, labels)
            
            # 计算检测到的目标数量
            detected_count = len(labels)
            
            results.append({
                "model": model,
                "image": img_name,
                "segment": segment,
                "detected_count": detected_count
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(export_dir, "vertebra_segment_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ 椎体节段分析结果已保存: {csv_path}")
    
    return df

# ================== 生成节段识别统计 ==================
def generate_segment_statistics(df):
    """生成椎体节段识别统计"""
    # 按模型和节段分组统计
    segment_stats = df.groupby(["model", "segment"]).agg({
        "detected_count": ["count", "mean", "std"]
    }).round(2)
    
    # 保存统计结果
    stats_path = os.path.join(export_dir, "segment_statistics.csv")
    segment_stats.to_csv(stats_path)
    print(f"✅ 节段统计结果已保存: {stats_path}")
    
    return segment_stats

# ================== 可视化节段识别结果 ==================
def visualize_segment_results(df):
    """可视化节段识别结果"""
    # 按模型和节段统计检测数量
    segment_counts = df.groupby(["model", "segment"]).size().unstack(fill_value=0)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    segment_counts.plot(kind="bar", stacked=True)
    plt.title("各模型在不同节段的检测数量", fontsize=14)
    plt.ylabel("检测数量")
    plt.xlabel("模型")
    plt.xticks(rotation=15)
    plt.legend(title="椎体节段")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    bar_path = os.path.join(export_dir, "segment_detection_counts.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"✅ 节段检测数量柱状图已保存: {bar_path}")
    
    # 绘制饼图（每个模型的节段分布）
    for model in models:
        model_df = df[df["model"] == model]
        if len(model_df) == 0:
            continue
        
        segment_dist = model_df["segment"].value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.pie(segment_dist.values, labels=segment_dist.index, autopct="%1.1f%%")
        plt.title(f"{model} 模型的节段分布", fontsize=14)
        plt.tight_layout()
        
        pie_path = os.path.join(export_dir, f"{model}_segment_distribution.png")
        plt.savefig(pie_path, dpi=300)
        plt.close()
        print(f"✅ {model} 节段分布饼图已保存: {pie_path}")

# ================== 主函数 ==================
def main():
    print("开始分析单椎体识别与定位结果...")
    
    # 分析椎体节段识别结果
    df = analyze_vertebra_segmentation()
    
    # 生成节段统计
    generate_segment_statistics(df)
    
    # 可视化结果
    visualize_segment_results(df)
    
    print("\n🎉 单椎体识别与定位结果分析完成！")

if __name__ == "__main__":
    main()
