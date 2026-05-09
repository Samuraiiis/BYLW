import os

def count_labeled_images(root_dir, image_exts=('.jpg', '.png', '.jpeg'), label_ext='.json'):
    print(f"\n统计标注进度：{root_dir}")
    print("-" * 60)
    total_imgs, total_labeled = 0, 0

    for dirpath, _, filenames in os.walk(root_dir):
        imgs = [f for f in filenames if f.lower().endswith(image_exts)]
        jsons = [f for f in filenames if f.lower().endswith(label_ext)]
        if imgs:
            labeled = 0
            for img in imgs:
                name = os.path.splitext(img)[0]
                if f"{name}{label_ext}" in jsons:
                    labeled += 1
            total_imgs += len(imgs)
            total_labeled += labeled
            print(f"{os.path.relpath(dirpath, root_dir):<45} | 图片: {len(imgs):>3} | 已标注: {labeled:>3} | 未标注: {len(imgs)-labeled:>3}")

    print("-" * 60)
    print(f"总计图片数: {total_imgs}, 已标注: {total_labeled}, 未标注: {total_imgs - total_labeled}")
    print(f"整体完成率: {total_labeled / total_imgs * 100:.1f}%\n")

if __name__ == "__main__":
    dataset_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    count_labeled_images(dataset_root)
