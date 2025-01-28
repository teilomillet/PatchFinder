"""Transformers backend for PatchFinder."""

import torch
import logging
from typing import Optional, Tuple
from PIL import Image
from ..core import PatchFinder
from ..confidence import calculate_patch_confidence

class TransformersPatchFinder(PatchFinder):
    """PatchFinder implementation for Hugging Face Transformers models."""
    
    def __init__(
        self,
        model,
        processor,
        patch_size: int = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        super().__init__(patch_size, overlap, logger, max_workers)
        self.model = model
        self.processor = processor
        self._configure_torch()

    def _configure_torch(self):
        torch.set_num_threads(self.max_workers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _process_patch(self, patch: Image, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            messages = [{
                "role": "user",
                "content": f"<|image_1|>\n{prompt}"
            }]
            
            formatted_prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            self.logger.debug(f"Formatted prompt:\n{formatted_prompt}")

            if patch.mode != 'RGB':
                patch = patch.convert('RGB')

            inputs = self.processor(
                text=formatted_prompt,
                images=patch,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                if self.model.generation_config.temperature != 0:
                    raise ValueError("Temperature must be 0 for confidence calculations")
                
                if self.model.generation_config.do_sample:
                    raise ValueError("Sampling must be disabled for confidence analysis")

                generate_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=0.0,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # Get generated text
            generated_ids = generate_ids.sequences[0, inputs["input_ids"].shape[1]:]
            text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate confidence
            logits = torch.stack(generate_ids.scores, dim=1).cpu().numpy()
            confidence = calculate_patch_confidence(logits)
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None 