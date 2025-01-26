# core.py
from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Union, Any
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
import torch

class PatchFinder(ABC):
    """Base class for PatchFinder implementations supporting multiple backends."""
    
    @classmethod
    def wrap(cls, model: Any, processor: Optional[Any] = None, **kwargs) -> 'PatchFinder':
        """Factory method to create appropriate PatchFinder instance based on model type.
        
        Args:
            model: The vision-language model instance
            processor: Optional processor/tokenizer for the model
            **kwargs: Additional arguments passed to PatchFinder constructor
        
        Returns:
            PatchFinder: Appropriate PatchFinder implementation
        """
        # Import here to avoid circular imports
        try:
            from vllm import LLM
            if isinstance(model, LLM):
                return VLLMPatchFinder(model, **kwargs)
        except ImportError:
            pass
            
        # Default to Transformers implementation
        return TransformersPatchFinder(model, processor, **kwargs)

    def __init__(
        self,
        patch_size: Union[int, float] = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        """Initialize base PatchFinder.
        
        Args:
            patch_size: Patch dimension (default: 256)
            overlap: Patch overlap ratio (default: 0.25)
            logger: Custom logger (optional)
            max_workers: Parallel processing threads (default: 1)
        """
        self._validate_params(patch_size, overlap)
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.logger = logger or self._configure_default_logger()

    def _validate_params(self, patch_size, overlap):
        if isinstance(patch_size, float):
            if not (0 < patch_size <= 1):
                raise ValueError(f"Invalid patch_size %: {patch_size}")
        elif isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"Invalid patch_size: {patch_size}")
        else:
            raise TypeError("patch_size must be int or float")
        if not (0 <= overlap < 1):
            raise ValueError(f"Invalid overlap: {overlap}")

    def _configure_default_logger(self):
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger

    @abstractmethod
    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[float]:
        """Process a single patch and return confidence score."""
        pass

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
                    "processed_patches": 0
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
                "processed_patches": 0
            }

    def _compile_results(self, confidences):
        avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
        return {
            "text": "",
            "confidence": round(avg_conf, 4),
            "processed_patches": len(confidences)
        }

class TransformersPatchFinder(PatchFinder):
    """PatchFinder implementation for Hugging Face Transformers models."""
    
    def __init__(
        self,
        model,
        processor,
        patch_size: Union[int, float] = 256,
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

    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[float]:
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

            logits = torch.stack(generate_ids.scores, dim=1).cpu().numpy()
            return calculate_patch_confidence(logits)
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None

class VLLMPatchFinder(PatchFinder):
    """PatchFinder implementation for vLLM models."""
    
    def __init__(
        self,
        llm,
        patch_size: Union[int, float] = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        super().__init__(patch_size, overlap, logger, max_workers)
        self.llm = llm
        
        # Verify vLLM config
        if not getattr(llm.sampling_params, "temperature", 0) == 0:
            raise ValueError("vLLM temperature must be 0 for confidence calculations")
        if not getattr(llm.sampling_params, "output_logprobs", True):
            raise ValueError("vLLM output_logprobs must be enabled")

    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[float]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # vLLM specific processing here
            # This is a placeholder - actual implementation would depend on vLLM's API
            outputs = self.llm.generate(prompt, images=[patch])
            
            # Extract logprobs from vLLM output
            logprobs = outputs[0].outputs[0].logprobs
            return calculate_patch_confidence(logprobs)
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None

# For backward compatibility
LegacyPatchFinder = TransformersPatchFinder