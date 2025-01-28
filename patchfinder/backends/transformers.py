"""Transformers backend for PatchFinder."""

import torch
import logging
from typing import Optional, Tuple
from PIL import Image
from ..core import PatchFinder
from ..confidence import calculate_patch_confidence

class TransformersPatchFinder(PatchFinder):
    """PatchFinder implementation for Transformers models."""
    
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
        
        # Validate model settings for confidence calculation
        if self.model.generation_config.temperature != 0:
            self.logger.warning("Setting temperature to 0 for confidence calculations")
            self.model.generation_config.temperature = 0.0
            
        if self.model.generation_config.do_sample:
            self.logger.warning("Disabling sampling for confidence analysis")
            self.model.generation_config.do_sample = False
            
    def _get_vocab_size(self) -> Optional[int]:
        """Extract vocabulary size from model configuration."""
        try:
            # Try model config first
            if hasattr(self.model, 'config'):
                vocab_size = getattr(self.model.config, 'vocab_size', None)
                if vocab_size:
                    return vocab_size
                    
            # Fallback to tokenizer vocab size
            if hasattr(self.processor, 'tokenizer'):
                vocab_size = len(self.processor.tokenizer)
                if vocab_size > 0:
                    return vocab_size
                    
            return None
        except Exception as e:
            self.logger.warning(f"Could not determine vocab size: {str(e)}")
            return None

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
            
            # Get logits and convert to numpy
            logits = torch.stack(generate_ids.scores, dim=1).cpu().numpy()
            
            # Get vocabulary size for proper normalization
            vocab_size = self._get_vocab_size()
            if vocab_size:
                self.logger.debug(f"Using vocabulary size: {vocab_size}")
            
            # Calculate confidence with vocab size if available
            confidence = calculate_patch_confidence(
                logprobs=logits,
                vocab_size=vocab_size
            )
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None 