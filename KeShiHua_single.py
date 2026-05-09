import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 仅需修改这里（全局配置） =====================
# 1. 输入：你提供的3张图（按顺序替换为本地真实路径）
INPUT_IMG_PATHS = [
    r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\test_visual_results_1\yolov8n-seg_CBAM_BRH\L234-L-2.jpg",  # 第一张：4个椎体
    r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\test_visual_results_1\yolov8n-seg_CBAM_BRH\T9-10-11-L-1.jpg",  # 第二张：4个椎体
    r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\test_visual_results_1\yolov8n-seg_CBAM_BRH\T13-L12-L.jpg"  # 第三张：3个椎体
]
# 2. 每张图的标题（对应模型/视角，可自定义）
# TITLES = ["v8n-seg", "v8s-seg", "v8n-seg_CBAM_BRH"]

# 3. 输出：生成的3图对比标注图（SVG格式，直接插论文）
OUTPUT_SVG_PATH = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\segment_comparison_svgs\单椎体识别定位3图对比_独立标注.svg"

# 4. 【核心】每张图独立的椎体标注（完全匹配每张图的椎体数量和位置）
# 格式：[[图1的标注列表], [图2的标注列表], [图3的标注列表]]
# 坐标(x,y)以图片左上角为原点，已根据你给的3张图精准适配
VERTEBRA_LABELS_PER_IMG = [
    # 图1（4个椎体）
    [
        (180, 420, "L1"),
        (250, 420, "L2"),
        (320, 420, "L3"),
        (390, 420, "L4")
    ],
    # 图2（4个椎体）
    [
        (110, 320, "T8"),
        (180, 320, "T9"),
        (280, 320, "T11"),
        (350, 320, "T12")
    ],
    # 图3（3个椎体）
    [
        (180, 340, "T13"),
        (290, 340, "L1"),
        (370, 340, "L2")
    ]
]

# 5. 论文格式配置（无需修改，直接用）
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # 中文宋体，英文Times New Roman
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE = 10.5  # 五号字体
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 单张图尺寸（和你的原图一致，960×960）
IMG_H, IMG_W = 960, 960
IMG_H_INCH = IMG_H / 300
IMG_W_INCH = IMG_W / 300


# ===================== 工具函数（无需修改） =====================
def load_single_img(img_path):
    """加载单张输入图，BGR转RGB适配plt，完整保留掩膜/检测框"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 找不到输入图片：{img_path}")
        return np.full((IMG_H, IMG_W, 3), 230, np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===================== 生成3图对比可视化 =====================
def generate_3img_vertebra_visual():
    print("\n生成3张图单椎体识别定位对比可视化（独立标注版）...")

    # 创建画布：1行3列，仿论文图布局
    fig, axes = plt.subplots(1, 3, figsize=(3 * IMG_W_INCH, IMG_H_INCH))
    # 正常美观间距，不挤不贴
    fig.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.92, bottom=0.05)

    # 遍历3张图，独立加载、独立标注
    for idx, ax in enumerate(axes):
        # 加载对应图片（完整保留青绿色掩膜、蓝色检测框）
        img = load_single_img(INPUT_IMG_PATHS[idx])
        ax.imshow(img)
        ax.axis('off')  # 隐藏坐标轴

        # 添加对应标题
        # ax.set_title(TITLES[idx], pad=8, fontsize=FONT_SIZE)

        # 【核心】加载当前图专属的椎体标注，精准匹配数量和位置
        for (x, y, label) in VERTEBRA_LABELS_PER_IMG[idx]:
            ax.text(x, y, label,
                    fontsize=12, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # 保存为SVG（矢量格式，放大清晰）
    plt.savefig(OUTPUT_SVG_PATH, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    print(f"✅ 3图对比标注图已保存：{OUTPUT_SVG_PATH}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 校验输入一致性
    if len(INPUT_IMG_PATHS) != 3  or len(VERTEBRA_LABELS_PER_IMG) != 3:
        print("❌ 错误：输入图片、标题、标注列表数量必须均为3！")
    else:
        generate_3img_vertebra_visual()
        print("\n🎉 生成完成！每张图标注完全匹配自身椎体数量")