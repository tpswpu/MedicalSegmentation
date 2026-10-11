# -*- coding: utf-8 -*-
"""
Unified Medical Knowledge Base Builder v2.0

This script builds a unified knowledge base in JSON format from multiple,
structurally different medical imaging datasets (MSD, ISIC, BraTS).

It is driven by a central configuration list (DATASET_CONFIGS) that defines
which datasets to process and how. For each image with a valid corresponding
label mask, it extracts:
1. The full path to the source image.
2. A predefined text description for that dataset/task.
3. A bounding box [x, y, width, height] calculated from the label mask.

The final output is a single JSON file ready for downstream tasks.
"""
import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# 配置区 (CONFIGURATION AREA)
# ==============================================================================

# 1. 数据集根目录
#    请确保这是您存放所有数据集（如MSD, ISIC2018, BraTS）的顶级目录
ROOT_DATA_DIR = r"D:\Datasets"

# 2. 数据集处理配置列表
#    在这里定义您想要处理的所有数据集。
#    - type: 数据集类型 ('msd', 'isic', 'brats')，用于调用专属处理器
#    - name: 自定义名称，用于日志打印
#    - path: 相对于ROOT_DATA_DIR的路径
#    - text_label: 用于该数据集中所有图像的固定文本描述
DATASET_CONFIGS = [
    # --- MSD 任务配置 ---
    {
        "type": "msd",
        "name": "MSD_Task02_Heart",
        "path": r"MSD\Task02_Heart",
        "text_label": "心脏影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task04_Hippocampus",
        "path": r"MSD\Task04_Hippocampus",
        "text_label": "海马体影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task05_Prostate",
        "path": r"MSD\Task05_Prostate",
        "text_label": "前列腺影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task09_Spleen",
        "path": r"MSD\Task09_Spleen",
        "text_label": "脾脏影像的切片"
    },
    # --- ISIC2018 数据集配置 ---
    {
        "type": "isic",
        "name": "ISIC2018",
        "path": r"ISIC2018",
        "text_label": "皮肤上的黑色斑块"
    },
    # --- BraTS 数据集配置 ---
    {
        "type": "brats",
        "name": "BraTS2021",
        "path": r"BraTS\BraTS2021",
        "text_label": "脑肿瘤影像"
    }
]

# 3. 定义最终输出的知识库文件路径
OUTPUT_JSON_PATH = "./unified_medical_kb.json"


# ==============================================================================
# 通用核心函数
# ==============================================================================

def calculate_bbox_from_mask(mask_path):
    """从一个二值掩码图片中计算边界框。"""
    try:
        with Image.open(mask_path) as mask_image:
            grayscale_mask = mask_image.convert('L')
            np_mask = np.array(grayscale_mask)

        rows, cols = np.where(np_mask > 0)

        if rows.size == 0:
            return None

        x_min, y_min = int(np.min(cols)), int(np.min(rows))
        x_max, y_max = int(np.max(cols)), int(np.max(rows))

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return [x_min, y_min, width, height]
    except Exception as e:
        tqdm.write(f"\n[警告] 处理掩码文件 {mask_path} 时发生错误: {e}")
        return None


# ==============================================================================
# 数据集专属处理器
# ==============================================================================

def process_msd_task(config):
    """处理单个MSD任务目录。"""
    task_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    images_dir = os.path.join(task_path, 'imagesTr')
    labels_dir = os.path.join(task_path, 'labelsTr')

    if not os.path.isdir(images_dir): return []

    image_paths = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, name))

    for image_path in tqdm(image_paths, desc=f"处理 {config['name']}", leave=False):
        label_path_str = image_path.replace('imagesTr', 'labelsTr')
        if os.path.exists(label_path_str):
            bbox = calculate_bbox_from_mask(label_path_str)
            if bbox:
                records.append({
                    "image_path": str(Path(image_path)),
                    "text": text_label,
                    "box": bbox
                })
    return records


def process_isic_dataset(config):
    """处理ISIC2018数据集。"""
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    for subset in ['train', 'test', 'validate']:
        # ★ 修改：根据您的描述，将 'images' 文件夹改为 'images_compressed'
        images_dir = os.path.join(dataset_path, subset, 'images_compressed')
        groundtruth_dir = os.path.join(dataset_path, subset, 'groundtruth')

        if not os.path.isdir(images_dir): continue

        for image_filename in tqdm(os.listdir(images_dir), desc=f"处理 {config['name']}/{subset}", leave=False):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            image_path = os.path.join(images_dir, image_filename)
            base_name = Path(image_filename).stem
            mask_filename = f"{base_name}.jpg"  # ISIC的文件名映射规则
            mask_path = os.path.join(groundtruth_dir, mask_filename)

            if os.path.exists(mask_path):
                bbox = calculate_bbox_from_mask(mask_path)
                if bbox:
                    records.append({
                        "image_path": str(Path(image_path)),
                        "text": text_label,
                        "box": bbox
                    })
    return records


def process_brats_dataset(config):
    """处理BraTS数据集。"""
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    images_dir = os.path.join(dataset_path, 'images')
    masks_dir = os.path.join(dataset_path, 'masks')

    if not os.path.isdir(images_dir): return []

    for image_filename in tqdm(os.listdir(images_dir), desc=f"处理 {config['name']}", leave=False):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        image_path = os.path.join(images_dir, image_filename)
        # 假设BraTS的mask和image文件名完全相同
        mask_path = os.path.join(masks_dir, image_filename)

        if os.path.exists(mask_path):
            bbox = calculate_bbox_from_mask(mask_path)
            if bbox:
                records.append({
                    "image_path": str(Path(image_path)),
                    "text": text_label,
                    "box": bbox
                })
    return records


# ==============================================================================
# 主调度函数
# ==============================================================================
def main():
    """主函数，作为总调度中心。"""
    print("======================================================")
    print("      多源医学影像知识库统一构建脚本 v2.0")
    print("======================================================")

    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"错误: 根目录 '{ROOT_DATA_DIR}' 不存在。请检查路径。")
        sys.exit(1)

    knowledge_base = {}  # ★ 修改：使用字典来存储，以ID为主键

    handler_map = {
        'msd': process_msd_task,
        'isic': process_isic_dataset,
        'brats': process_brats_dataset
    }

    # 主循环：遍历配置，调用相应处理器
    for config in DATASET_CONFIGS:
        dataset_type = config.get('type')
        if dataset_type in handler_map:
            print(f"\n>>>>> 开始处理数据集类型: '{dataset_type.upper()}', 名称: '{config['name']}' <<<<<")
            new_records = handler_map[dataset_type](config)

            # ★ 修改：将记录添加到字典中
            for record in new_records:
                # 使用 image_path 作为唯一ID
                record_id = str(Path(record["image_path"]).as_posix())
                knowledge_base[record_id] = record

            print(f">>>>> '{config['name']}' 处理完成，更新/新增 {len(new_records)} 条记录。 <<<<<")
        else:
            print(f"\n警告: 未知的处理器类型 '{dataset_type}'，跳过 '{config['name']}'。")

    # 最终报告与保存
    print("\n======================================================")
    print("所有数据集处理完毕！")
    print(f"  - 成功生成知识库总记录数: {len(knowledge_base)}")

    output_file = Path(OUTPUT_JSON_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n正在将统一知识库保存到: {output_file}...")
    try:
        # ★ 修改：将字典转换为列表进行保存，以保持与旧格式的兼容性
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(knowledge_base.values()), f, indent=2, ensure_ascii=False)
        print(f"'{output_file}' 保存成功！")
    except Exception as e:
        print(f"!!!!!! 保存JSON文件时发生严重错误: {e}")

    print("======================================================")


if __name__ == "__main__":
    main()
