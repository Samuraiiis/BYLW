import os
import cv2
import numpy as np
from glob import glob

# ================== 基本配置 ==================
project_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
base_dir = os.path.join(project_root, "test_visual_results")

# 行（从上到下）的模型顺序——按你想看的顺序写
models = [
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8n-seg_CBAM_BRH",
    "yolov7",
]

# 参考模型（用它的文件名作为列对齐基准；也可改成 'yolov7'）
ref_model = "yolov8n-seg"

# 选取多少列图片（从参考模型按文件名排序后取前N张）
num_cols = 4  # 想要几列就改这里

# 每个图像单元格的显示高度（会等比缩放）
cell_img_h = 400  # 增加图像高度
# 左侧模型名栏宽度
label_col_w = 350  # 增加模型名栏宽度
# 网格线宽度和颜色
grid_th = 3  # 增加网格线宽度
grid_color = (100, 100, 100)  # 加深网格线颜色
# 背景色（浅灰）
bg_color = (240, 240, 240)

# 输出路径
out_dir = os.path.join(base_dir, "comparison_matrix")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"matrix_{ref_model}_{num_cols}cols.jpg")


# ================== 工具函数 ==================
def list_images(folder):
    return sorted(glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png")))

def put_text_center(img, text, font_scale=1.0, color=(0, 0, 0), thickness=3):
    """在矩形区域中心写字（水平垂直尽量居中）"""
    (w, h) = (img.shape[1], img.shape[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max((w - tw) // 2, 5)
    y = max((h + th) // 2, th + 5)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def load_and_fit(path, target_h):
    """读取并按高度等比缩放到 target_h；失败则返回占位灰图"""
    if not os.path.exists(path):
        # 占位：浅灰底，写 Missing
        dummy = np.full((target_h, target_h, 3), 230, np.uint8)
        put_text_center(dummy, "Missing", 1.0, (60,60,60), 3)
        return dummy
    img = cv2.imread(path)
    if img is None:
        dummy = np.full((target_h, target_h, 3), 230, np.uint8)
        put_text_center(dummy, "ReadErr", 1.0, (60,60,60), 3)
        return dummy
    h, w = img.shape[:2]
    new_w = int(w * (target_h / h))
    return cv2.resize(img, (new_w, target_h))


# ================== 组装列名（从参考模型拿） ==================
# ================== 组装列名（手动选择） ==================
# 你可以手动列出想拼接的图片名（必须与模型结果目录下的文件名一致）
col_names = [
    "0003_2 (2).jpg",
    "0004_2 (3).jpg",
    "0012_5 (2).jpg",
    "0016_6 (3).jpg",
    "0041_5.jpg"
]
num_cols = len(col_names)


# ================== 预读取所有单元格图像并确定每列的统一宽 ==================
# 为了矩阵整齐：同一列的图像统一宽度（取该列所有行中最大宽）
col_max_w = [0]*num_cols
cells = []  # cells[row][col] = 图像
for r, m in enumerate(models):
    row_cells = []
    m_dir = os.path.join(base_dir, m)
    for c, name in enumerate(col_names):
        path = os.path.join(m_dir, name)
        cell_img = load_and_fit(path, cell_img_h)
        row_cells.append(cell_img)
        col_max_w[c] = max(col_max_w[c], cell_img.shape[1])
    cells.append(row_cells)

# ================== 计算画布大小并创建背景 ==================
# 每个模型名单元格高度 = cell_img_h
rows = len(models)
cols = num_cols + 1  # +1 是左侧模型名列
# 每行高度相同：cell_img_h
row_h = cell_img_h
# 每列宽度：第一列=label_col_w；其他列=col_max_w[c-1]
col_ws = [label_col_w] + col_max_w

canvas_h = rows * row_h + (rows + 1) * grid_th
canvas_w = sum(col_ws) + (cols + 1) * grid_th
canvas = np.full((canvas_h, canvas_w, 3), bg_color, np.uint8)

# ================== 逐格绘制（含网格线、文字、图像） ==================
y = grid_th
for r in range(rows):
    x = grid_th
    # 1) 左侧模型名单元格
    cell = np.full((row_h, col_ws[0], 3), 255, np.uint8)
    put_text_center(cell, models[r], font_scale=0.7, color=(0,0,0), thickness=2)
    canvas[y:y+row_h, x:x+col_ws[0]] = cell
    x += col_ws[0] + grid_th

    # 2) 图片单元格
    for c in range(num_cols):
        w = col_ws[c+1]
        # 把真实图像按“居中”贴入固定宽度的白底单元格
        cell = np.full((row_h, w, 3), 255, np.uint8)
        img = cells[r][c]
        ih, iw = img.shape[:2]
        off_x = max((w - iw)//2, 0)
        cell[:, off_x:off_x+iw] = img[:, :min(iw, w)]
        canvas[y:y+row_h, x:x+w] = cell
        x += w + grid_th

    y += row_h + grid_th

# 画外框和内部网格线
# 横线
y = 0
for r in range(rows+1):
    cv2.rectangle(canvas, (0, y), (canvas_w-1, y+grid_th-1), grid_color, -1)
    y += row_h + grid_th
# 竖线
x = 0
for c in range(cols+1):
    cv2.rectangle(canvas, (x, 0), (x+grid_th-1, canvas_h-1), grid_color, -1)
    x += col_ws[c if c < len(col_ws) else -1] + grid_th

# 可选：在每列顶部加“Image1 / 文件名”等标题（此处改为文件名）
# 这里不单独加标题行，如果你想要，可在最上方再加一行 header。

# ================== 保存 ==================
cv2.imwrite(out_path, canvas)
print(f"✅ 矩阵图已保存：{out_path}")
