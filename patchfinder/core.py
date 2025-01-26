# core.py
from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Union, Any, Tuple
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
import torch
import numpy as np

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
            
        # Check for MLX model
        if "mlx" in str(type(model)):
            try:
                import mlx.core as mx
                return MLXPatchFinder(model, processor, **kwargs)
            except ImportError:
                raise ImportError("MLX backend detected but mlx package not installed. Please install mlx-lm.")
            
        # Default to Transformers implementation
        return TransformersPatchFinder(model, processor, **kwargs)

    def __init__(
        self,
        patch_size: Union[int, float] = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1,
        aggregation_mode: str = "max"
    ):
        """Initialize base PatchFinder.
        
        Args:
            patch_size: Patch dimension (default: 256)
            overlap: Patch overlap ratio (default: 0.25)
            logger: Custom logger (optional)
            max_workers: Parallel processing threads (default: 1)
            aggregation_mode: How to aggregate patch confidences ('max', 'min', 'average')
        """
        self._validate_params(patch_size, overlap)
        self._validate_aggregation_mode(aggregation_mode)
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.logger = logger or self._configure_default_logger()
        self.aggregation_mode = aggregation_mode.lower()

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

    def _validate_aggregation_mode(self, mode: str):
        """Validate the aggregation mode."""
        valid_modes = {"max", "min", "average"}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid aggregation_mode: {mode}. "
                f"Valid options: {valid_modes}"
            )

    def _configure_default_logger(self):
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger

    @abstractmethod
    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        """Process a single patch and return text and confidence score."""
        pass

    def extract(
        self,
        image_path: str,
        prompt: str = "Extract text",
        timeout: int = 30,
        aggregation_mode: Optional[str] = None
    ) -> Dict:
        """Enhanced extraction with deadlock prevention and configurable confidence aggregation.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for extraction
            timeout: Maximum time per patch in seconds
            aggregation_mode: Override instance aggregation mode
        """
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

            results = []
            for idx, patch in enumerate(patches):
                result = self._process_patch(patch, prompt, timeout, idx)
                if result is not None:
                    text, confidence = result
                    results.append((text, confidence))

            # Use override mode if provided, else instance mode
            mode = aggregation_mode.lower() if aggregation_mode else self.aggregation_mode
            return self._compile_results(results, mode)
        
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {
                "text": "", 
                "confidence": 0.0,
                "processed_patches": 0
            }

    def _compile_results(self, results, mode: str):
        """Compile results using specified aggregation mode."""
        if not results:
            return {
                "text": "",
                "confidence": 0.0,
                "processed_patches": 0
            }
            
        texts, confidences = zip(*results)
        confidences = np.array(confidences)
        
        # Calculate aggregate confidence
        if mode == "max":
            best_idx = np.argmax(confidences)
            confidence = confidences[best_idx]
            selected_text = texts[best_idx]
        elif mode == "min":
            confidence = np.min(confidences)
            # Still use highest confidence text
            best_idx = np.argmax(confidences)
            selected_text = texts[best_idx]
        else:  # average
            confidence = np.mean(confidences)
            # Still use highest confidence text
            best_idx = np.argmax(confidences)
            selected_text = texts[best_idx]
        
        return {
            "text": selected_text,
            "confidence": round(float(confidence), 4),
            "processed_patches": len(results)
        }

class MLXPatchFinder(PatchFinder):
    """PatchFinder implementation for MLX models."""
    
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
        
    def _mlx_generate_with_logprobs(self, prompt_tokens, patch, max_tokens=500):
        """Generate text and compute logprobs using MLX's generate_step."""
        import mlx.core as mx
        
        self.logger.debug("Starting MLX generation")
        detokenizer = self.processor.tokenizer.detokenizer
        detokenizer.reset()
        all_logprobs = []
        
        try:
            for (token, logprobs), _ in zip(
                self.model.generate_step(prompt_tokens, temperature=0.0),
                range(max_tokens)
            ):
                if token == self.processor.tokenizer.eos_token_id:
                    break
                    
                detokenizer.add_token(token)
                all_logprobs.append(logprobs)
                
            text = detokenizer.text
            logprobs_array = np.array([lp.tolist() for lp in all_logprobs])
            
            return text, logprobs_array
            
        except Exception as e:
            self.logger.error(f"MLX generation failed: {str(e)}", exc_info=True)
            return None, None

    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # Format prompt
            messages = [{
                "role": "user",
                "content": f"<image>\n{prompt}"
            }]
            
            formatted_prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Convert patch if needed
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
                
            # Process inputs
            inputs = self.processor(
                text=formatted_prompt,
                images=[patch],
                return_tensors="np"
            )
            
            # Generate with logprobs
            text, logprobs = self._mlx_generate_with_logprobs(
                inputs["input_ids"][0],
                patch
            )
            
            if text is None or logprobs is None:
                return None
                
            # Calculate confidence
            confidence = calculate_patch_confidence(logprobs)
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None

# Update existing implementations to match new interface

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

    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
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

    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # vLLM specific processing here
            outputs = self.llm.generate(prompt, images=[patch])
            
            # Extract text and logprobs
            text = outputs[0].outputs[0].text
            logprobs = outputs[0].outputs[0].logprobs
            confidence = calculate_patch_confidence(logprobs)
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None

# For backward compatibility
LegacyPatchFinder = TransformersPatchFinder