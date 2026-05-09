import os, json
import numpy as np
from PIL import Image

# 类别名称映射（确保与 data.yaml 一致）
classes = ["spinous_process", "supraspinous_ligament"]

def convert_json(json_path, save_txt=True):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_h, img_w = data["imageHeight"], data["imageWidth"]
    txt_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in classes:
            continue
        cls_id = classes.index(label)

        points = np.array(shape["points"], dtype=float)
        # 坐标归一化
        points[:, 0] /= img_w
        points[:, 1] /= img_h
        points = points.reshape(-1).tolist()

        txt_line = f"{cls_id} " + " ".join([f"{p:.6f}" for p in points])
        txt_lines.append(txt_line)

    if save_txt:
        txt_path = json_path.replace(".json", ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(txt_lines))

def batch_convert(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".json"):
                convert_json(os.path.join(dirpath, file))

if __name__ == "__main__":
    dataset_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    batch_convert(dataset_root)
    print("✅ 所有 Labelme 标注已转换为 YOLOv8 segmentation 格式！")
