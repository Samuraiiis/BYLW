import os

def count_images_in_folders(root_dir, extensions=('.jpg', '.png', '.jpeg')):
    print(f"\n统计路径: {root_dir}")
    print("--------------------------------------------------")
    total_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 统计每个文件夹中的图片数量
        img_files = [f for f in filenames if f.lower().endswith(extensions)]
        count = len(img_files)
        if count > 0:
            total_count += count
            relative_path = os.path.relpath(dirpath, root_dir)
            print(f"{relative_path:<40} : {count:>4} 张图片")
    print("--------------------------------------------------")
    print(f"总计图片数: {total_count} 张\n")

# 示例：替换为你的数据路径
if __name__ == "__main__":
    dataset_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
    count_images_in_folders(dataset_root)
