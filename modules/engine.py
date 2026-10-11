import logging
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.V5.configs.prompts import ULTIMATE_MCOT_PROMPT_TEMPLATE, ZERO_SHOT_PROMPT_TEMPLATE


class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        engine_config = config['model_configs']['reasoning_engine']
        model_path = engine_config['model_path']
        logging.info("Initializing Reasoning Engine...")
        logging.info(f" - Loading Qwen2.5-VL model from '{model_path}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            load_in_8bit=engine_config.get('use_int8', False),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if engine_config.get('use_flash_attention_2', False) else "eager",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        logging.info("✅ Reasoning Engine initialized successfully.")

    def _parse_box_from_string(self, raw_output: str) -> Optional[List[int]]:
        """Robustly extracts a bounding box from the VLM's text output."""
        match = re.search(r'(\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*)', raw_output)
        if match:
            try:
                # Normalize comma variants before splitting.
                cleaned_str = match.group(1).replace('，', ',')
                return [int(float(n.strip())) for n in cleaned_str.split(',')]
            except (ValueError, IndexError):
                return None
        return None

    def _run_vlm(self, prompt: str, images: List[Image.Image]) -> Optional[List[int]]:
        """Generic VLM execution function."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)

        raw_output = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        logging.info(f"--- [VLM Raw Output] ---\n{raw_output}\n------------------------")

        return self._parse_box_from_string(raw_output)

    def run_ultimate_mcot(self, image: Image.Image, text_prompt: str, reference_example: Dict) -> Optional[List[int]]:
        """Executes MCoT reasoning with a reference example."""
        logging.info("Executing contextual VLM reasoning (MCoT)...")
        ref_image = Image.open(reference_example['data']['image_path']).convert("RGB")
        prompt = ULTIMATE_MCOT_PROMPT_TEMPLATE.format(
            text_prompt=text_prompt,
            rag_box=reference_example['data']['box']
        )
        return self._run_vlm(prompt, [image, ref_image])

    def run_zero_shot_vlm(self, image: Image.Image, text_prompt: str) -> Optional[List[int]]:
        """Executes zero-shot VLM localization without a reference example."""
        logging.info("Executing zero-shot VLM localization...")
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text_prompt=text_prompt)
        return self._run_vlm(prompt, [image])
