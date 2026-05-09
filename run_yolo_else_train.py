import os
import subprocess
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ========== 基础路径（与你之前一致） ==========
PROJECT_ROOT = r"E:\PyCharm 2023.3.3\Projects\EI_PROJ"
DATA_YAML    = os.path.join(PROJECT_ROOT, "data.yaml")
RESULT_ROOT  = os.path.join(PROJECT_ROOT, "train_results")      # 训练/评估统一放这里
COMPARE_DIR  = os.path.join(RESULT_ROOT, "comparison")          # 汇总/图表
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

# 你已训练好的 v8 模型（请按实际路径改）
V8_BASELINE_BEST = os.path.join(PROJECT_ROOT, r"runs_seg\yolov8n-seg\weights\best.pt")
V8_CUSTOM_BEST   = os.path.join(PROJECT_ROOT, r"runs_seg\yolov8n-seg2\weights\best.pt")

# v5/v7 预训练权重（放在根目录）
V5_WEIGHT = os.path.join(PROJECT_ROOT, "yolov5s-seg.pt")
V7_WEIGHT = os.path.join(PROJECT_ROOT, "yolov7-seg.pt")

# 源码目录（必须存在）
V5_DIR = os.path.join(PROJECT_ROOT, "yolov5")
V7_DIR = os.path.join(PROJECT_ROOT, "yolov7")

def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_cmd(cmd, cwd):
    if not os.path.isdir(cwd):
        raise NotADirectoryError(f"工作目录不存在: {cwd}")
    print(f"\n>>> 执行: {cmd}\n")
    p = subprocess.Popen(cmd, shell=True, cwd=cwd,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {cmd}")

# ========== 训练 YOLOv5n-seg ==========
def train_yolov5s_seg():
    save_dir = os.path.join(RESULT_ROOT, f"yolov5s-seg_{ts()}")
    os.makedirs(save_dir, exist_ok=True)
    cmd = (
        f"python train.py "
        f"--img 512 --batch 10 --epochs 100 "
        f"--data \"{DATA_YAML}\" "
        f"--weights \"{V5_WEIGHT}\" "
        f"--project \"{save_dir}\" "
        f"--name yolov5n-seg_train "
        f"--device 0"
    )

    run_cmd(cmd, cwd=V5_DIR)
    print(f"YOLOv5s-seg 训练结果: {save_dir}")
    return save_dir

# ========== 训练 YOLOv7-seg ==========
def train_yolov7_seg():
    save_dir = os.path.join(RESULT_ROOT, f"yolov7-seg_{ts()}")
    os.makedirs(save_dir, exist_ok=True)
    # yolov7 的分割训练脚本与参数在不同fork可能略有差异，以下为通用写法
    cmd = (
        f"python train.py "
        f"--img 512 --batch-size 10 --epochs 100 "
        f"--data \"{DATA_YAML}\" "
        f"--weights \"{V7_WEIGHT}\" "
        f"--project \"{save_dir}\" "
        f"--name yolov7-seg_train "
        f"--device 0"
    )

    run_cmd(cmd, cwd=V7_DIR)
    print(f"YOLOv7-seg 训练结果: {save_dir}")
    return save_dir

# ========== 评估：v5/v7 用各自仓库，v8 用 ultralytics ==========
def eval_v5_on_test(best_weight_path):
    # yolov5 的 val.py 支持分割模型，会根据权重自动识别任务
    out_dir = os.path.join(COMPARE_DIR, "YOLOv5s-seg_metrics")
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"python val.py "
        f"--weights {best_weight_path} "
        f"--data {DATA_YAML} "
        f"--img 512 --task test "
        f"--project {out_dir} --name metrics --save-json"
    )
    run_cmd(cmd, cwd=V5_DIR)
    return os.path.join(out_dir, "metrics", "results.csv")

def eval_v7_on_test(best_weight_path):
    # 不同分支中可能是 test.py 或 val.py，这里先尝试 val.py，不行再用 test.py
    out_dir = os.path.join(COMPARE_DIR, "YOLOv7-seg_metrics")
    os.makedirs(out_dir, exist_ok=True)
    tried = []
    try:
        cmd = (
            f"python val.py "
            f"--weights {best_weight_path} "
            f"--data {DATA_YAML} "
            f"--img 512 "
            f"--project {out_dir} --name metrics"
        )
        run_cmd(cmd, cwd=V7_DIR)
        tried.append("val.py")
    except Exception:
        cmd = (
            f"python test.py "
            f"--weights {best_weight_path} "
            f"--data {DATA_YAML} "
            f"--img-size 512 "
            f"--project {out_dir} --name metrics"
        )
        run_cmd(cmd, cwd=V7_DIR)
        tried.append("test.py")
    print(f"YOLOv7 使用评估脚本: {tried[-1]}")
    # 结果文件名在不同分支可能不同，这里优先找 results.csv
    cand = [
        os.path.join(out_dir, "metrics", "results.csv"),
        os.path.join(out_dir, "metrics", "results.txt")
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return None

def eval_v8_on_test(best_weight_path, tag_name):
    from ultralytics import YOLO
    out_dir = os.path.join(COMPARE_DIR, f"{tag_name}_metrics")
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(best_weight_path)
    results = model.val(data=DATA_YAML, imgsz=512, split="test", project=out_dir, name="metrics")
    # ultralytics 会在 out_dir/metrics 下生成 results.csv
    csv_path = os.path.join(out_dir, "metrics", "results.csv")
    return csv_path if os.path.exists(csv_path) else None

# ========== 从训练目录里寻找 best.pt ==========
def find_best_weight(train_save_dir):
    # 兼容 yolov5/yolov7 的保存结构：.../weights/best.pt
    w1 = os.path.join(train_save_dir, "yolov5n-seg_train", "weights", "best.pt")
    w2 = os.path.join(train_save_dir, "yolov7-seg_train", "weights", "best.pt")
    w3 = os.path.join(train_save_dir, "weights", "best.pt")
    for w in [w1, w2, w3]:
        if os.path.exists(w):
            return w
    # 回退：递归搜索
    for r, d, f in os.walk(train_save_dir):
        if "best.pt" in f:
            return os.path.join(r, "best.pt")
    return None

# ========== 解析/统一指标 ==========
def parse_results(csv_path):
    """
    返回统一键：precision, recall, f1, map50, map5095, iou(若无就不返回)
    不同框架的列名可能不同，这里做尽量兼容。
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    # 读取最后一行（总体 all）
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    # 取汇总（通常最后一行或包含 'all' 的行）
    row = df.iloc[-1]
    txtcols = [c for c in df.columns if isinstance(c, str)]
    # 映射候选列名
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return float(row[c])
        return None

    precision = pick('precision', 'P', 'box_precision', 'seg_precision')
    recall    = pick('recall', 'R', 'box_recall', 'seg_recall')
    map50     = pick('map50', 'mAP_0.5', 'box_map50', 'seg_map50')
    map5095   = pick('map', 'map50-95', 'mAP_0.5:0.95', 'box_map', 'seg_map')
    iou       = pick('iou', 'IoU', 'mask_iou', 'seg_iou')

    # 计算 F1（若 P/R 存在）
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    out = {}
    if precision is not None: out['precision'] = precision
    if recall    is not None: out['recall']    = recall
    if f1        is not None: out['f1']        = f1
    if map50     is not None: out['mAP@0.5']   = map50
    if map5095   is not None: out['mAP@0.5-0.95'] = map5095
    if iou       is not None: out['IoU']       = iou
    return out

# ========== 画柱状图 ==========
def plot_bar(metrics_dict, save_path):
    if not metrics_dict:
        return
    df = pd.DataFrame(metrics_dict).T  # 模型为行
    # 只画存在的列
    cols_order = [c for c in ['precision','recall','f1','mAP@0.5','mAP@0.5-0.95','IoU'] if c in df.columns]
    df = df[cols_order]
    ax = df.plot(kind="bar", figsize=(10,6), rot=0, title="Model Performance Comparison (Test Set)")
    ax.set_ylabel("Score")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"对比图已保存: {save_path}")

# ========== 主流程 ==========
if __name__ == "__main__":
    print("=== 开始训练 YOLOv5s-seg 与 YOLOv7-seg，并与 YOLOv8 两个版本进行测试集对比 ===")
    print(f"数据集: {DATA_YAML}")
    print(f"输出根目录: {RESULT_ROOT}")

    # 1) 训练 v5 / v7
    if not os.path.exists(V5_WEIGHT):
        raise FileNotFoundError(f"未找到 yolov5s-seg 权重: {V5_WEIGHT}")
    if not os.path.exists(V7_WEIGHT):
        raise FileNotFoundError(f"未找到 yolov7-seg 权重: {V7_WEIGHT}")
    if not os.path.isdir(V5_DIR) or not os.path.isdir(V7_DIR):
        raise NotADirectoryError("请确认 yolov5/ 与 yolov7/ 源码目录已存在于项目根目录。")

    v5_train_dir = train_yolov5n_seg()
    v7_train_dir = train_yolov7_seg()

    v5_best = find_best_weight(v5_train_dir)
    v7_best = find_best_weight(v7_train_dir)
    if not v5_best:
        raise FileNotFoundError("未找到 YOLOv5 训练生成的 best.pt")
    if not v7_best:
        raise FileNotFoundError("未找到 YOLOv7 训练生成的 best.pt")

    # 2) 在测试集统一评估四个模型
    print("\n=== 测试集评估 ===")
    v5_csv = eval_v5_on_test(v5_best)
    v7_csv = eval_v7_on_test(v7_best)

    if not os.path.exists(V8_BASELINE_BEST):
        print(f"⚠️ 未找到 v8 baseline: {V8_BASELINE_BEST}（将跳过）")
    if not os.path.exists(V8_CUSTOM_BEST):
        print(f"⚠️ 未找到 v8 自定义: {V8_CUSTOM_BEST}（将跳过）")

    v8_base_csv  = eval_v8_on_test(V8_BASELINE_BEST, "YOLOv8n-seg") if os.path.exists(V8_BASELINE_BEST) else None
    v8_cust_csv  = eval_v8_on_test(V8_CUSTOM_BEST,   "YOLOv8n-seg_CBAM_BRH") if os.path.exists(V8_CUSTOM_BEST) else None

    # 3) 解析并汇总
    summary = {}
    summary["YOLOv5s-seg"]            = parse_results(v5_csv)
    summary["YOLOv7-seg"]             = parse_results(v7_csv)
    if v8_base_csv: summary["YOLOv8n-seg"]           = parse_results(v8_base_csv)
    if v8_cust_csv: summary["YOLOv8n-seg+CBAM+BRH"]  = parse_results(v8_cust_csv)

    # 保存表格 & 图
    df_out = pd.DataFrame(summary).T
    csv_out = os.path.join(COMPARE_DIR, "metrics_comparison.csv")
    df_out.to_csv(csv_out, float_format="%.4f")
    print(f"\n指标汇总已保存: {csv_out}")
    plot_bar(summary, os.path.join(COMPARE_DIR, "metrics_bar.png"))

    print("\n✅ 全流程完成！请在以下位置查看：")
    print(f"- 训练输出：{RESULT_ROOT}")
    print(f"- 对比表格：{csv_out}")
    print(f"- 对比图：  {os.path.join(COMPARE_DIR, 'metrics_bar.png')}")
