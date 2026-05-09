import os
import cv2
import numpy as np
from glob import glob

# ==== 基础路径 ====
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
base_dir = os.path.join(project_root, "test_visual_results")
output_dir = os.path.join(base_dir, "comparison_side")
os.makedirs(output_dir, exist_ok=True)

# ==== 模型目录 ====
models = [
    "yolov7",
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8n-seg_CBAM_BRH",
]

# ==== 以第一个模型为基准 ====
ref_dir = os.path.join(base_dir, models[0])
ref_imgs = sorted(glob(os.path.join(ref_dir, "*.jpg")) + glob(os.path.join(ref_dir, "*.png")))
print(f"✅ 找到 {len(ref_imgs)} 张图片用于拼接。")

# ==== 拼接 ====
for i, ref_path in enumerate(ref_imgs, 1):
    name = os.path.basename(ref_path)
    imgs = []
    valid_models = []
    from difflib import get_close_matches

    for m in models:
        model_dir = os.path.join(base_dir, m)
        all_files = [os.path.basename(f) for f in glob(os.path.join(model_dir, "*"))]
        all_lower = [f.lower().strip() for f in all_files]
        target = name.lower().strip()

        match_path = None
        # ✅ 1. 精确匹配
        if target in all_lower:
            match_path = os.path.join(model_dir, all_files[all_lower.index(target)])
        else:
            # ✅ 2. 模糊匹配（允许括号、空格不同）
            candidates = [f for f in all_files if
                          f.lower().replace(" ", "").replace("(", "").replace(")", "").startswith(
                              target.replace(" ", "").replace("(", "").replace(")", ""))]
            if candidates:
                match_path = os.path.join(model_dir, candidates[0])
            else:
                # ✅ 3. 最后尝试相似度匹配
                close = get_close_matches(target, all_lower, n=1, cutoff=0.8)
                if close:
                    match_path = os.path.join(model_dir, all_files[all_lower.index(close[0])])

        if not match_path or not os.path.exists(match_path):
            print(f"⚠️ {m} 实际存在但匹配失败: {name}")
            continue

        img = cv2.imread(match_path)
        if img is None:
            print(f"⚠️ {m} 读取失败: {match_path}")
            continue

        imgs.append(img)
        valid_models.append(m)

    if len(imgs) == 0:
        continue

    # ==== 统一高度 ====
    h_min = min(im.shape[0] for im in imgs)
    imgs_resized = [cv2.resize(im, (int(im.shape[1]*h_min/im.shape[0]), h_min)) for im in imgs]

    # ==== 在模型间加入分隔线 ====
    line_color = (0, 0, 0)     # 黑线，可改为 (200,200,200) 灰色
    line_thickness = 4         # 线宽
    sep_imgs = []
    for idx, img in enumerate(imgs_resized):
        sep_imgs.append(img)
        if idx < len(imgs_resized) - 1:
            sep = np.ones((h_min, line_thickness, 3), dtype=np.uint8) * 255
            sep[:] = line_color
            sep_imgs.append(sep)
    concat = cv2.hconcat(sep_imgs)

    # ==== 顶部标签 ====
    label_h = 60  # 增加标签高度
    canvas = np.ones((concat.shape[0] + label_h, concat.shape[1], 3), dtype=np.uint8) * 255
    canvas[label_h:, :, :] = concat

    step = concat.shape[1] // len(valid_models)
    for idx, m in enumerate(valid_models):
        text_x = idx * step + step // 2
        # 计算文本宽度以居中
        text_size = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = text_x - text_size[0] // 2
        cv2.putText(canvas, m, (max(text_x, 10), 45),  # 调整垂直位置
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # 增加字体大小和粗细

    # ==== 保存 ====
    save_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_concat.jpg")
    cv2.imwrite(save_path, canvas)
    print(f"[{i}/{len(ref_imgs)}] ✅ 保存对比图: {save_path}")

print(f"\n🎉 所有拼接图已保存到: {output_dir}")
