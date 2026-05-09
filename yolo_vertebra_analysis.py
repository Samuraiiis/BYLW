import os
from glob import glob

# ================== 基本配置 ==================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
runs_dir = os.path.join(project_root, "runs_seg")
export_dir = os.path.join(project_root, "vertebra_analysis")
os.makedirs(export_dir, exist_ok=True)

# ================== 模型列表 ==================
models = [
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov7",
]

# ================== 读取 YOLO 预测结果 ==================
def read_yolo_predictions(pred_dir):
    """读取 YOLO 模型的预测结果"""
    predictions = []
    
    # 获取所有预测图像和标签
    pred_images = glob(os.path.join(pred_dir, "*.jpg")) + glob(os.path.join(pred_dir, "*.png"))
    
    for img_path in pred_images:
        img_name = os.path.basename(img_path)
        
        # 查找对应的标签文件
        label_path = os.path.join(pred_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        # 读取标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        confidence = float(parts[5]) if len(parts) > 5 else 1.0
                        labels.append([class_id, confidence])
        
        predictions.append({
            "image": img_name,
            "labels": labels,
            "label_count": len(labels)
        })
    
    return predictions

# ================== 识别椎体节段 ==================
def identify_vertebra_segment(image_name):
    """根据图像名称识别椎体节段"""
    # 从图像名称中提取节段信息
    # 例如：0003_2 (2).jpg 中的 2 可能代表 L2
    # 这里需要根据实际的命名规则进行调整
    
    # 简单规则：如果图像名称包含数字，提取第一个数字作为节段标识
    import re
    digits = re.findall(r'\d+', image_name)
    if digits:
        # 假设第一个数字代表节段
        seg_num = digits[0]
        # 映射数字到节段，例如 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=L6, 7=L7, 8=T8, 9=T9, 10=T10
        seg_map = {
            "1": "L1",
            "2": "L2",
            "3": "L3",
            "4": "L4",
            "5": "L5",
            "6": "L6",
            "7": "L7",
            "8": "T8",
            "9": "T9",
            "10": "T10",
            "11": "T11",
            "12": "T12"
        }
        return seg_map.get(seg_num, "Unknown")
    return "Unknown"

# ================== 分析模型预测结果 ==================
def analyze_model_predictions():
    """分析模型的预测结果"""
    all_results = []
    
    for model in models:
        model_dir = os.path.join(runs_dir, model)
        if not os.path.exists(model_dir):
            print(f"⚠️ 模型目录不存在: {model_dir}")
            continue
        
        # 查找预测结果目录
        pred_dirs = [
            os.path.join(model_dir, "predictions"),
            os.path.join(model_dir, "test"),
            os.path.join(model_dir, "val")
        ]
        
        pred_dir = None
        for d in pred_dirs:
            if os.path.exists(d):
                pred_dir = d
                break
        
        if not pred_dir:
            print(f"⚠️ 未找到预测结果目录: {model}")
            continue
        
        # 读取预测结果
        predictions = read_yolo_predictions(pred_dir)
        
        # 分析每个预测结果
        for pred in predictions:
            # 识别椎体节段
            segment = identify_vertebra_segment(pred["image"])
            
            # 统计不同类别的检测数量
            class_counts = {}
            for label in pred["labels"]:
                class_id = label[0]
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
            
            all_results.append({
                "model": model,
                "image": pred["image"],
                "segment": segment,
                "total_detections": pred["label_count"],
                "class_counts": class_counts
            })
    
    # 保存结果
    output_path = os.path.join(export_dir, "yolo_predictions_analysis.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("模型,图像,节段,总检测数,类别分布\n")
        for result in all_results:
            class_dist = str(result["class_counts"])
            f.write(f"{result['model']},{result['image']},{result['segment']},{result['total_detections']},{class_dist}\n")
    
    print(f"✅ YOLO 预测结果分析已保存: {output_path}")
    return all_results

# ================== 生成模型节段性能统计 ==================
def generate_model_segment_stats(results):
    """生成模型在不同节段上的性能统计"""
    # 按模型和节段分组
    model_segment_stats = {}
    
    for result in results:
        model = result["model"]
        segment = result["segment"]
        
        if model not in model_segment_stats:
            model_segment_stats[model] = {}
        
        if segment not in model_segment_stats[model]:
            model_segment_stats[model][segment] = {
                "total_images": 0,
                "total_detections": 0,
                "avg_detections": 0
            }
        
        model_segment_stats[model][segment]["total_images"] += 1
        model_segment_stats[model][segment]["total_detections"] += result["total_detections"]
    
    # 计算平均值
    for model in model_segment_stats:
        for segment in model_segment_stats[model]:
            stats = model_segment_stats[model][segment]
            stats["avg_detections"] = stats["total_detections"] / stats["total_images"] if stats["total_images"] > 0 else 0
    
    # 保存统计结果
    stats_path = os.path.join(export_dir, "model_segment_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("模型,节段,图像数量,总检测数,平均检测数\n")
        for model in model_segment_stats:
            for segment in model_segment_stats[model]:
                stats = model_segment_stats[model][segment]
                f.write(f"{model},{segment},{stats['total_images']},{stats['total_detections']},{stats['avg_detections']:.2f}\n")
    
    print(f"✅ 模型节段性能统计已保存: {stats_path}")
    return model_segment_stats

# ================== 主函数 ==================
def main():
    print("开始分析 YOLO 模型的单椎体识别与定位结果...")
    
    # 分析模型预测结果
    results = analyze_model_predictions()
    
    # 生成模型节段性能统计
    generate_model_segment_stats(results)
    
    print("\n🎉 YOLO 模型单椎体识别与定位结果分析完成！")

if __name__ == "__main__":
    main()
