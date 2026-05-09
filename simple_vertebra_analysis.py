import os
import json
from glob import glob

# ================== 基本配置 ==================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
test_dir = os.path.join(project_root, "test")
export_dir = os.path.join(project_root, "vertebra_analysis")
os.makedirs(export_dir, exist_ok=True)

# ================== 读取标签文件 ==================
def read_yolo_labels(label_path):
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

def read_json_labels(json_path):
    """读取JSON格式的标签文件"""
    labels = []
    if not os.path.exists(json_path):
        return labels
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'shapes' in data:
                for shape in data['shapes']:
                    class_id = 0  # 默认为0，实际应该根据类别映射
                    labels.append([class_id, 0, 0, 0, 0])  # 简化处理
    except Exception as e:
        print(f"读取JSON文件失败: {json_path}, 错误: {e}")
    return labels

# ================== 分析测试数据 ==================
def analyze_test_data():
    """分析测试数据中的椎体节段"""
    results = []
    
    # 遍历测试目录
    for view_dir in os.listdir(test_dir):
        view_path = os.path.join(test_dir, view_dir)
        if not os.path.isdir(view_path):
            continue
        
        for seg_dir in os.listdir(view_path):
            seg_path = os.path.join(view_path, seg_dir)
            if not os.path.isdir(seg_path):
                continue
            
            # 获取该节段的所有图像
            images = glob(os.path.join(seg_path, "*.jpg")) + glob(os.path.join(seg_path, "*.png"))
            
            for img_path in images:
                img_name = os.path.basename(img_path)
                
                # 查找对应的标签文件
                txt_path = os.path.join(seg_path, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
                json_path = os.path.join(seg_path, img_name.replace(".jpg", ".json").replace(".png", ".json"))
                
                # 读取标签
                labels = []
                if os.path.exists(txt_path):
                    labels = read_yolo_labels(txt_path)
                elif os.path.exists(json_path):
                    labels = read_json_labels(json_path)
                
                # 记录结果
                results.append({
                    "view": view_dir,
                    "segment": seg_dir,
                    "image": img_name,
                    "label_count": len(labels)
                })
    
    # 保存结果
    output_path = os.path.join(export_dir, "test_data_analysis.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("视图,节段,图像,标签数量\n")
        for result in results:
            f.write(f"{result['view']},{result['segment']},{result['image']},{result['label_count']}\n")
    
    print(f"✅ 测试数据分析结果已保存: {output_path}")
    return results

# ================== 生成节段统计 ==================
def generate_statistics(results):
    """生成节段统计"""
    # 按节段统计
    segment_stats = {}
    for result in results:
        segment = result['segment']
        if segment not in segment_stats:
            segment_stats[segment] = {
                'total_images': 0,
                'total_labels': 0
            }
        segment_stats[segment]['total_images'] += 1
        segment_stats[segment]['total_labels'] += result['label_count']
    
    # 保存统计结果
    stats_path = os.path.join(export_dir, "segment_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("节段,图像数量,标签总数,平均标签数\n")
        for segment, stats in segment_stats.items():
            avg_labels = stats['total_labels'] / stats['total_images'] if stats['total_images'] > 0 else 0
            f.write(f"{segment},{stats['total_images']},{stats['total_labels']},{avg_labels:.2f}\n")
    
    print(f"✅ 节段统计结果已保存: {stats_path}")
    return segment_stats

# ================== 主函数 ==================
def main():
    print("开始分析单椎体识别与定位结果...")
    
    # 分析测试数据
    results = analyze_test_data()
    
    # 生成统计
    generate_statistics(results)
    
    print("\n🎉 单椎体识别与定位结果分析完成！")

if __name__ == "__main__":
    main()
