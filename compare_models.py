from ultralytics import YOLO
import os

# 数据配置文件
data_yaml = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\data.yaml"

# 模型路径字典（包含你自己的模型）
models = {
    # "YOLOv5s-seg": "yolov5s-seg.pt",
    "YOLOv6n-seg": "yolov6n_seg.pt",
    "YOLOv7-seg": "yolov7-seg.pt",
    "YOLOv8n-seg": "yolov8n-seg.pt",
    "Ours_YOLOv8n_CBAM_BRH": r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\runs_seg\yolov8n-seg2\weights\best.pt"
}

# 数据集
test_source = "test_1"  # 测试集目录

# 保存结果的根目录
save_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\compare_test_results"
os.makedirs(save_root, exist_ok=True)

for name, model_path in models.items():
    print(f"\n🚀 开始评估模型: {name}")
    model = YOLO(model_path)

    # 1️⃣ 在测试集上计算性能指标
    eval_dir = os.path.join(save_root, name, "metrics")
    os.makedirs(eval_dir, exist_ok=True)

    results = model.val(
        data=data_yaml,
        imgsz=512,
        split="test",  # ✅ 改为测试集
        project=eval_dir,
        name="metrics",
        save_json=True
    )

    print(f"✅ {name} 测试集评估完成：mAP@0.5={results.box.map50:.3f}, mAP@0.5–0.95={results.box.map:.3f}")

    # 2️⃣ 生成测试集预测图（用于论文可视化）
    pred_dir = os.path.join(save_root, name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    model.predict(
        source=test_source,
        imgsz=512,
        save=True,
        boxes=False,
        retina_masks=True,
        show_labels=False,
        show_conf=False,
        project=pred_dir
    )

    print(f"🎨 {name} 测试集预测图已保存至: {pred_dir}")

print("\n✅ 所有模型在测试集上的评估与可视化已完成。")
