# 2_build_feature_matrix.py
# -*- coding: utf-8 -*-
"""
Feature Matrix Builder for Brute-force Search v1.0

This script reads a unified knowledge base, extracts image features using
a BLIP model, and saves them as a NumPy array (.npz file) for brute-force
similarity search.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 配置区 (CONFIGURATION AREA)
# ==============================================================================
KNOWLEDGE_BASE_PATH = "./unified_medical_kb.json"
OUTPUT_DIR = "./scripts/outputs_medical/"
MODEL_NAME = "blip_feature_extractor"
MODEL_TYPE = "base"


def build_feature_matrix(kb_path, output_dir, model, vis_processors, device):
    knowledge_base_path = Path(kb_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ★ 输出文件现在是一个.npz文件，包含特征和ID
    features_output_path = output_path / "medical_features.npz"

    print(f"从 '{knowledge_base_path}' 加载知识库...")
    if not knowledge_base_path.exists():
        print(f"错误: 知识库文件 '{knowledge_base_path}' 未找到。")
        sys.exit(1)

    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    all_features = []
    valid_kb_ids = []

    print(f"开始为 {len(knowledge_base)} 个条目计算特征...")
    for entry in tqdm(knowledge_base, desc="提取图像特征"):
        try:
            image_path = entry["image_path"]
            raw_image = Image.open(image_path).convert("RGB")

            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_feature = model.extract_features(
                    {"image": image}, mode="image"
                ).image_embeds_proj[:, 0, :].squeeze()

            # ★ 关键：对特征向量进行归一化，为后续的内积计算做准备
            image_feature /= image_feature.norm()

            all_features.append(image_feature.cpu().numpy())
            # 将图片的唯一路径作为ID记录下来
            valid_kb_ids.append(str(Path(image_path).as_posix()))

        except Exception as e:
            tqdm.write(f"\n警告: 处理路径: {entry.get('image_path')} 时发生错误，已跳过。错误: {e}")
            continue

    if not all_features:
        print("\n错误: 未能从知识库中成功提取任何特征。")
        sys.exit(1)

    features_matrix = np.array(all_features, dtype=np.float32)

    print(f"特征提取完成，共 {features_matrix.shape[0]} 个有效特征。")
    print(f"正在将特征矩阵和ID保存到: {features_output_path} ...")
    np.savez(features_output_path, features=features_matrix, ids=np.array(valid_kb_ids))
    print("✅ 特征矩阵保存成功！")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======================================================")
    print("        特征矩阵构建脚本 v1.0 (暴力搜索版)")
    print("======================================================")
    print(f"使用设备: {device}")

    print(f"\n正在加载特征提取器: {MODEL_NAME} (类型: {MODEL_TYPE})...")
    model, vis_processors, _ = load_model_and_preprocess(
        name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=True, device=device
    )
    print("模型加载成功！")

    build_feature_matrix(
        kb_path=KNOWLEDGE_BASE_PATH,
        output_dir=OUTPUT_DIR,
        model=model,
        vis_processors=vis_processors,
        device=device
    )
    print("\n所有任务完成！")