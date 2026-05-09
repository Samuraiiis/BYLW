import os
import glob
import subprocess
from ultralytics import YOLO

# =========================================================
# 路径配置
# =========================================================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
# test_dir = os.path.join(project_root, "test_flat")
test_dir = os.path.join(project_root, "test_1")
base_dir = os.path.join(project_root, "runs_seg")
output_root = os.path.join(project_root, "test_visual_results_1")
os.makedirs(output_root, exist_ok=True)

# =========================================================
# 收集所有测试图片
# =========================================================
test_source = glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
test_source += glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True)
print(f"✅ 找到 {len(test_source)} 张测试图片")

if len(test_source) == 0:
    raise RuntimeError(f"❌ 未在 {test_dir} 下找到任何图片，请检查路径。")

# YOLOv7 的 detect.py 只支持 txt 输入，因此写入文件
test_list_path = os.path.join(project_root, "test_list.txt")
with open(test_list_path, "w", encoding="utf-8") as f:
    f.write("\n".join(test_source))

# =========================================================
# 搜索所有 best.pt 模型
# =========================================================
models = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f == "best.pt":
            models.append(os.path.join(root, f))

if not models:
    raise FileNotFoundError("❌ 未在 runs_seg 下找到任何 best.pt，请确认模型训练完成。")

# =========================================================
# 自动检查 detect.py 是否支持 view_img 参数
# =========================================================
def detect_has_view_img(detect_path: str) -> bool:
    """判断 YOLOv7 detect.py 是否包含 view_img 参数"""
    try:
        with open(detect_path, "r", encoding="utf-8") as f:
            content = f.read()
        return "--view-img" in content or "view_img" in content
    except Exception:
        return False


# =========================================================
# 模型推理主循环
# =========================================================
for m in models:
    model_name = os.path.basename(os.path.dirname(os.path.dirname(m)))
    print(f"\n🚀 开始生成预测结果: {model_name}")

    save_dir = os.path.join(output_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # =====================================================
    # YOLOv8 分支（分割）
    # =====================================================
    if "yolov8" in model_name.lower():
        print("⚙️ YOLOv8 模型 (分割) 推理中 ...")

        model = YOLO(m)
        model.predict(
            source=test_source,
            save=True,
            boxes=False,          # ✅ 保留检测框
            show_labels=False,   # ❌ 不显示类别名
            show_conf=False,     # ❌ 不显示置信度蓝条
            retina_masks=True,   # ✅ 高分辨率掩膜
            project=output_root,
            name=model_name,
            exist_ok=True
        )

    # =====================================================
    # YOLOv7 分支（检测）
    # =====================================================
    elif "yolov7" in model_name.lower():
        print("⚙️ YOLOv7 模型使用 detect.py 进行推理 ...")

        yolov7_root = os.path.join(project_root, "yolov7")
        detect_script = os.path.join(yolov7_root, "detect.py")

        has_view = detect_has_view_img(detect_script)
        if has_view:
            print("🧩 检测到 detect.py 含 view_img 参数，自动禁用窗口显示。")

        # ✅ 递归查找 test 文件夹下所有图片
        all_imgs = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            all_imgs.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
        all_imgs = sorted(all_imgs)
        print(f"✅ 共找到 {len(all_imgs)} 张测试图片（含子文件夹）")

        # ✅ 写入临时 txt 列表（仅一次性使用）
        tmp_txt = os.path.join(project_root, "test_list_temp.txt")
        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(all_imgs))

        # ✅ 运行 detect.py
        cmd = [
            "python", detect_script,
            "--weights", m,
            "--source", test_dir,  # 使用 test_flat文件夹
            "--img-size", "512",
            "--conf-thres", "0.25",
            "--iou-thres", "0.45",
            "--device", "0",
            "--project", output_root,
            "--name", model_name,
            "--exist-ok",
        ]

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        subprocess.run(cmd, cwd=yolov7_root, check=True)

        # ✅ 推理完成后删除临时文件
        if os.path.exists(tmp_txt):
            os.remove(tmp_txt)

        print(f"✅ YOLOv7 模型 {model_name} 推理完成，结果已保存至: {output_root}\\{model_name}")


    # =====================================================
    # 未识别模型
    # =====================================================
    else:
        print(f"⚠️ 未识别模型类型: {model_name}，跳过。")

print("\n🎉 所有模型推理完成！结果已保存到：", output_root)
