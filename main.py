import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

V5_DIR = Path(__file__).resolve().parent
if str(V5_DIR) not in sys.path:
    sys.path.append(str(V5_DIR))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from modules.engine import ReasoningEngine
from modules.retriever import EfficientRetriever



def main():
    with open(V5_DIR / 'configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    log_path = Path(config['data_paths']['log_file'])
    setup_logger(log_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"--- Starting new batch segmentation and evaluation task ---")
    logging.info(f"Device: {DEVICE}")

    paths_config = config['data_paths']
    model_configs = config['model_configs']
    eval_configs = config['evaluation_configs']
    retrieval_configs = config['retrieval_configs']
    logic_thresholds = config['logic_thresholds']

    encoder_model, vis_processors, _ = load_model_and_preprocess(
        name=model_configs['encoder'], model_type=model_configs['encoder_model_type'],
        is_eval=True, device=DEVICE
    )
    retriever = EfficientRetriever(
        feature_matrix_path=paths_config['feature_matrix_path'],
        knowledge_base_path=paths_config['knowledge_base_json'],
        model=encoder_model, vis_processors=vis_processors, device=DEVICE
    )
    engine = ReasoningEngine(config=config)

    sam2_predictor = None
    sam2_configs = model_configs.get('sam2')
    if sam2_configs and sam2_configs.get('root_dir'):
        original_cwd = os.getcwd()
        sam2_root = Path(sam2_configs['root_dir'])
        try:
            logging.info(f"Temporarily changing working directory to: {sam2_root}")
            os.chdir(sam2_root)
            sam2_model = build_sam2(sam2_configs['model_cfg'], sam2_configs['checkpoint'])
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            sam2_predictor.model.to(DEVICE)
            logging.info("✅ SAM 2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SAM 2 model: {e}")
            sam2_predictor = None
        finally:
            os.chdir(original_cwd)
            logging.info(f"Restored working directory to: {original_cwd}")

    if not sam2_predictor:
        logging.error("SAM 2 model failed to load, terminating program.")
        return

    input_dir = Path(paths_config['input_image_dir'])
    output_dir = Path(paths_config['output_mask_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    shared_query_text = "Lesions on the skin"
    logging.info(f"Found {len(image_files)} images in '{input_dir}', using uniform instruction: '{shared_query_text}'")

    visualization_list = set()
    if eval_configs.get('enable_visualization', False):
        sample_ratio = eval_configs.get('visualization_sample_ratio', 1.0)
        num_to_visualize = int(len(image_files) * sample_ratio)
        if num_to_visualize > 0:
            num_to_visualize = min(num_to_visualize, len(image_files))
            visualization_list = set(random.sample(image_files, num_to_visualize))
        logging.info(f"Visualization enabled, will randomly sample {len(visualization_list)} images out of {len(image_files)} for visualization.")

    all_results = []
    total_processing_time = 0.0
    processed_image_count = 0

    progress_bar = tqdm(image_files, desc="Batch Segmenting (Brute-force Search)")
    for image_path in progress_bar:
        start_time = time.time()

        result_entry = {"file_name": image_path.name, "query": shared_query_text}
        final_box = None
        try:
            query_image = Image.open(image_path).convert("RGB")
            query_image_size = query_image.size

            retrieved_results = retriever.retrieve(
                query_image, shared_query_text, top_k=retrieval_configs.get('top_k', 5),
                image_weight=retrieval_configs['image_text_weight'],
                size_weight=retrieval_configs.get('size_similarity_weight', 0.0)
            )

            if not retrieved_results:
                final_path_msg = "Zero-Shot VLM (Retrieval Failed)"
                result_entry['final_path'] = final_path_msg
                progress_bar.write(f"File '{image_path.name}': RAG retrieval failed. -> {final_path_msg}")
                logging.warning(f"File '{image_path.name}': RAG retrieval failed. -> {final_path_msg}")
                final_box = engine.run_zero_shot_vlm(query_image, shared_query_text)
            else:
                best_reference = retrieved_results[0]
                result_entry['retrieval'] = best_reference
                similarity_score = best_reference['similarity_score']

                log_msg = (
                    f"File '{image_path.name}': "
                    f"Retrieved best candidate (final_score: {best_reference['final_score']:.4f}), "
                    f"content similarity: {similarity_score:.4f}"
                )
                progress_bar.write(log_msg)
                logging.info(log_msg)

                if similarity_score >= logic_thresholds['tier1_direct_mapping_similarity']:
                    result_entry['final_path'] = "Tier 1: Direct Mapping"
                    ref_box_abs = best_reference['data']['box']
                    with Image.open(best_reference['data']['image_path']) as ref_img:
                        ref_img_size = ref_img.size
                    ref_box_rel = box_abs_to_rel(ref_box_abs, ref_img_size)
                    final_box = box_rel_to_abs(ref_box_rel, query_image_size)
                elif similarity_score >= logic_thresholds['tier2_fallback_similarity']:
                    result_entry['final_path'] = "Tier 2: Robust Average Fallback"
                    final_box = generate_box_from_robust_average(retrieved_results, query_image_size)
                else:
                    result_entry['final_path'] = "Tier 3: Zero-Shot VLM (Low Confidence)"
                    final_box = engine.run_zero_shot_vlm(query_image, shared_query_text)

            if final_box:
                expansion_ratio = logic_thresholds.get('box_expansion_ratio', 1.0)
                if expansion_ratio > 1.0:
                    logging.info(f"  -> Expanding Box by a ratio of {expansion_ratio}...")
                    final_box = expand_box(final_box, query_image_size, expansion_ratio)

                result_entry['final_box'] = final_box

                x, y, w, h = final_box
                input_box_xyxy = np.array([x, y, x + w, y + h])
                center_x, center_y = x + w / 2, y + h / 2
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])

                logging.info(f"  -> Preparing dual Box and Point prompts for SAM 2...")

                with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                    sam2_predictor.set_image(query_image)
                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=input_box_xyxy,
                        multimask_output=True
                    )

                best_mask = masks[np.argmax(scores)]
                result_entry['final_mask'] = best_mask
                result_entry['mask_score'] = float(np.max(scores))

                output_mask_path = output_dir / image_path.name
                save_binary_mask(best_mask, output_mask_path)
            else:
                result_entry['error'] = "Failed to generate a box for SAM2."

            if eval_configs.get('enable_visualization',
                                False) and image_path in visualization_list and 'error' not in result_entry:
                progress_bar.write(f"  -> Generating visualization for '{image_path.name}'...")
                viz_save_dir = Path(paths_config['output_dir']) / "visualizations/batch_flow/"
                viz_save_dir.mkdir(exist_ok=True, parents=True)
                safe_filename = "".join(c for c in image_path.stem if c.isalnum()).rstrip()
                viz_output_path = viz_save_dir / f"{safe_filename}_viz.png"
                visualize_and_save_result(query_image, result_entry, viz_output_path)

        except Exception as e:
            logging.exception(f"A critical error occurred while processing file '{image_path.name}': {e}")
            result_entry['error'] = str(e)

        all_results.append(result_entry)

        end_time = time.time()
        total_processing_time += (end_time - start_time)
        processed_image_count += 1

    logging.info(">>> All images processed, starting final evaluation... <<<")

    gt_dir = paths_config.get('ground_truth_mask_dir')
    if gt_dir and Path(gt_dir).exists():
        avg_dice, avg_miou = calculate_metrics_for_folder(str(output_dir), gt_dir)
    else:
        logging.warning("ground_truth_mask_dir not provided or path does not exist in config, skipping evaluation.")
        avg_dice, avg_miou = -1.0, -1.0

    avg_time_per_image = total_processing_time / processed_image_count if processed_image_count > 0 else 0
    report_content = (
        f"Evaluation Metrics Report\n"
        f"=========================================\n"
        f"Total images processed: {processed_image_count}\n"
        f"Total processing time: {total_processing_time:.2f} seconds\n"
        f"Average time per image: {avg_time_per_image:.3f} seconds\n"
        f"-----------------------------------------\n"
        f"Dice Score: {avg_dice:.4f}\n"
        f"Mean IoU (mIoU): {avg_miou:.4f}\n"
        f"=========================================\n"
    )

    print("\n" + report_content)
    metrics_file = Path(paths_config['metrics_output_file'])
    metrics_file.parent.mkdir(exist_ok=True, parents=True)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logging.info(f"✅ Evaluation report saved to: {metrics_file}")
    logging.info("✅ All tasks completed.")


if __name__ == '__main__':
    main()