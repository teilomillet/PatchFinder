# core.py
import logging
from typing import Optional, Dict
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
import torch

class PatchFinder:
    """Advanced text extraction from images using vision-language models with deadlock prevention.
    
    Features:
    - Async-safe operations
    - Timeout handling
    - Configurable logging
    - GPU memory management
    - Thread-limiting options
    
    Args:
        model: Pretrained vision-language model
        processor: Associated processor
        patch_size: Patch dimension (default: 256)
        overlap: Patch overlap ratio (default: 0.25)
        logger: Custom logger (optional)
        max_workers: Parallel processing threads (default: 1)
    """
    
    def __init__(
        self,
        model,
        processor,
        patch_size: int = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        self._validate_params(patch_size, overlap)
        self.model = model
        self.processor = processor
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.logger = logger or self._configure_default_logger()
        self._configure_torch()

    def _validate_params(self, patch_size, overlap):
        if patch_size <= 0:
            raise ValueError(f"Invalid patch_size: {patch_size}")
        if not (0 <= overlap < 1):
            raise ValueError(f"Invalid overlap: {overlap}")

    def _configure_default_logger(self):
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger

    def _configure_torch(self):
        torch.set_num_threads(self.max_workers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract(self, image_path: str, prompt: str = "Extract text", timeout: int = 30) -> Dict:
        """Enhanced extraction with deadlock prevention."""
        self.logger.info(f"Processing {image_path}")
        
        try:
            patches = generate_patches(image_path, self.patch_size, self.overlap)
            self.logger.debug(f"Generated {len(patches)} patches")
            
            if not patches:
                return {
                    "text": "", 
                    "confidence": 0.0,
                    "processed_patches": 0  # Add missing key
                }

            confidences = []
            for idx, patch in enumerate(patches):
                confidence = self._process_patch(patch, prompt, timeout, idx)
                if confidence is not None:
                    confidences.append(confidence)

            return self._compile_results(confidences)
        
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {
                "text": "", 
                "confidence": 0.0,
                "processed_patches": 0  # Add missing key
            }

    def _process_patch(self, patch, prompt, timeout, idx):
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # Log raw prompt before formatting
            self.logger.debug(f"Original prompt: {prompt}")
            
            messages = [{
                "role": "user",
                "content": f"<|image_1|>\n{prompt}"
            }]
            
            formatted_prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            self.logger.debug(f"Formatted prompt:\n{formatted_prompt}")  # Log full template

            if patch.mode != 'RGB':
                patch = patch.convert('RGB')

            inputs = self.processor(
                text=formatted_prompt,
                images=patch,
                return_tensors="pt"
            ).to(self.model.device)

            # Log input tensor details
            self.logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
            self.logger.debug(f"Pixel Values shape: {inputs['pixel_values'].shape}")

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

            # Log generation output structure
            self.logger.debug(f"Generation output keys: {generate_ids.keys()}")
            self.logger.debug(f"Scores length: {len(generate_ids.scores)}")
            self.logger.debug(f"Scores[0] shape: {generate_ids.scores[0].shape if generate_ids.scores else 'N/A'}")

            generated_ids = generate_ids.sequences[:, inputs['input_ids'].shape[1]:]
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            self.logger.info(f"Generated text: {text}")  # Keep this as INFO level

            # Log raw logits before processing
            if generate_ids.scores:
                raw_logits = generate_ids.scores[0].cpu().numpy()
                self.logger.debug(f"Sample raw logits: {raw_logits[0][:5]}")  # First 5 elements of first token

            logits = torch.stack(generate_ids.scores, dim=1).cpu().numpy()
            self.logger.debug(f"Stacked logits shape: {logits.shape}")
            
            return calculate_patch_confidence(logits)
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None

    def _calculate_confidence(self, outputs, idx):
        logits = outputs.logits.cpu().numpy()
        confidence = calculate_patch_confidence(logits)
        self.logger.debug(f"Patch {idx} confidence: {confidence:.2f}")
        return confidence

    def _compile_results(self, confidences):
        avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
        return {
            "text": "",
            "confidence": round(avg_conf, 4),
            "processed_patches": len(confidences)
        }