import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
PROJECT_ROOT = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
TEST_DIR = os.path.join(PROJECT_ROOT, "test_visual_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "segment_comparison_svgs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE = 10.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ===================== 模型配置 =====================
V8_FOLDERS = ["YOLOv8n-seg", "YOLOv8s-seg", "YOLOv8n-seg_CBAM_BRH"]
V7_FOLDER = "YOLOv7"

V8_DISPLAY_NAMES = ["v8n-seg", "v8s-seg", "v8n-seg_CBAM_BRH"]
V7_DISPLAY_NAME = "v7"

V8_IMG_NAMES = ["0003_2 (2).jpg", "0016_6 (3).jpg"]
V7_IMG_NAME = "0012_5 (2).jpg"

V8_OUTPUT_SVG = os.path.join(OUTPUT_DIR, "v8模型对比图.svg")
V7_OUTPUT_SVG = os.path.join(OUTPUT_DIR, "v7单独效果图_局部放大.svg")

IMG_H_INCH = 480 / 300
IMG_W_INCH = 640 / 300


# ===================== 加载图片 =====================
def load_img(folder_name, img_name):
    img_path = os.path.join(TEST_DIR, folder_name, img_name)
    img_path = os.path.normpath(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 找不到：{img_path}")
        img = np.full((480, 640, 3), 230, np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===================== V8 对比图 =====================
def generate_v8_internal_comparison():
    print("\n生成 V8 对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(3 * IMG_W_INCH, 2 * IMG_H_INCH))
    fig.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.05)

    for row_idx in range(2):
        for col_idx in range(3):
            img = load_img(V8_FOLDERS[col_idx], V8_IMG_NAMES[row_idx])
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(V8_DISPLAY_NAMES[col_idx], pad=5, fontsize=FONT_SIZE)

    plt.savefig(V8_OUTPUT_SVG, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()


# ===================== V7 单独图 + 局部放大（高清检测文字） =====================
def generate_v7_single_with_zoom():
    print("\n生成 V7 单独图 + 局部放大（高清版）...")

    img = load_img(V7_FOLDER, V7_IMG_NAME)
    h, w = img.shape[:2]

    # ===================== 局部放大区域（你可以自己改坐标） =====================
    # 这里默认放大【中间区域】，你想放大哪里我帮你调
    x1, y1 = int(w * 0.17), int(h * 0.35)  # 左上角
    x2, y2 = int(w * 0.8), int(h * 0.8)  # 右下角
    zoom_img = img[y1:y2, x1:x2]

    # ===================== 绘图：原图 + 放大小窗口 =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IMG_W_INCH * 1.8, IMG_H_INCH))
    fig.subplots_adjust(wspace=0.1)

    # 左图：原图
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(V7_DISPLAY_NAME, fontsize=11)
    # 画框标出放大区域
    ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                edgecolor='red', linewidth=2, fill=False))

    # 右图：局部放大（这里的字会变大！）
    ax2.imshow(zoom_img)
    ax2.axis('off')
    ax2.set_title('局部放大', fontsize=11)

    plt.savefig(V7_OUTPUT_SVG, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    print(f"✅ V7 高清放大图已保存：{V7_OUTPUT_SVG}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    generate_v8_internal_comparison()
    generate_v7_single_with_zoom()
    print("\n🎉 全部生成完成！")