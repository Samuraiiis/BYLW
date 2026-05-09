import os, glob, shutil

src_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\test"       # 原 test 根目录（含子目录）
dst_root = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ\test_flat"  # 新平铺目录
os.makedirs(dst_root, exist_ok=True)

count = 0
for ext in ["*.jpg", "*.png", "*.jpeg"]:
    for f in glob.glob(os.path.join(src_root, "**", ext), recursive=True):
        name = os.path.basename(f)
        # 避免文件重名，加上索引
        new_name = f"{count:04d}_{name}"
        shutil.copy(f, os.path.join(dst_root, new_name))
        count += 1

print(f"✅ 已复制 {count} 张图片到 {dst_root}")
