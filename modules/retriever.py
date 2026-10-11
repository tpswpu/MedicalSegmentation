import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


class EfficientRetriever:
    def __init__(self, feature_matrix_path: str, knowledge_base_path: str, model, vis_processors, device: str):
        logging.info("Initializing brute-force retriever...")

        logging.info(f" - Loading feature matrix from '{feature_matrix_path}'...")
        feature_data = np.load(feature_matrix_path)
        self.features_matrix = feature_data['features']
        self.valid_kb_ids = feature_data['ids'].tolist()

        logging.info(f" - Loading knowledge base from '{knowledge_base_path}'...")
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            knowledge_base_list = json.load(f)
            self.kb_lookup = {str(Path(entry["image_path"]).as_posix()): entry for entry in knowledge_base_list}

        self.model = model
        self.vis_processors = vis_processors
        self.device = device
        logging.info("✅ Brute-force retriever initialized successfully.")

    def _encode_query(self, image: Image, text: str, image_weight: float) -> np.ndarray:
        image_processed = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        sample = {"image": image_processed, "text_input": [text]}
        with torch.no_grad():
            image_output = self.model.extract_features(sample, mode="image")
            image_features = image_output.image_embeds_proj[:, 0, :]

            text_output = self.model.extract_features(sample, mode="text")
            text_features = text_output.text_embeds_proj[:, 0, :]

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            multimodal_features = image_weight * image_features + (1.0 - image_weight) * text_features

        multimodal_features /= multimodal_features.norm(dim=-1, keepdim=True)
        return multimodal_features.cpu().numpy().astype('float32')

    def retrieve(
            self, image: Image, text: str, top_k: int = 1, image_weight: float = 0.7, size_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        query_vector = self._encode_query(image, text, image_weight)
        query_img_size = image.size

        similarities = np.dot(self.features_matrix, query_vector.T).squeeze()
        sorted_indices = np.argsort(similarities)[::-1]

        ranked_results = []
        for idx in sorted_indices:
            if similarities[idx] > 0.9999:
                 continue

            kb_id = self.valid_kb_ids[idx]
            entry = self.kb_lookup.get(kb_id)
            if not entry:
                continue

            try:
                ref_img_path = Path(entry['image_path'])
                with Image.open(ref_img_path) as ref_img:
                    ref_img_size = ref_img.size
                query_area = query_img_size[0] * query_img_size[1]
                ref_area = ref_img_size[0] * ref_img_size[1]
                size_similarity = min(query_area, ref_area) / max(query_area, ref_area) if max(query_area,
                                                                                               ref_area) > 0 else 0
                size_distance = 1.0 - size_similarity
            except Exception as e:
                logging.warning(f"Error calculating size similarity: {e}, this factor will be ignored.")
                size_distance = 0.5

            content_score = similarities[idx]
            content_distance = 1.0 - content_score
            final_score = content_distance + size_weight * size_distance

            ranked_results.append({
                "similarity_score": float(content_score),
                "size_distance": float(size_distance),
                "final_score": float(final_score),
                "data": entry
            })

            if len(ranked_results) >= (top_k + 5):
                break

        ranked_results.sort(key=lambda x: x['final_score'])

        return ranked_results[:top_k]