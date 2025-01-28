"""Core PatchFinder functionality."""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Union, Any, Tuple
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from PIL import Image

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
                from .backends import VLLMPatchFinder
                return VLLMPatchFinder(model, **kwargs)
        except ImportError:
            pass
            
        # Check for MLX model
        if "mlx" in str(type(model)):
            try:
                import mlx.core as mx
                from .backends import MLXPatchFinder
                return MLXPatchFinder(model, processor, **kwargs)
            except ImportError:
                raise ImportError("MLX backend detected but mlx package not installed. Please install mlx-lm.")
            
        # Default to Transformers implementation
        from .backends import TransformersPatchFinder
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
    def _process_patch(self, patch: Image, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        """Process a single patch and return text and confidence score."""
        pass

    def extract(
        self,
        image_path: str,
        prompt: str = "Extract text",
        timeout: int = 30,
        aggregation_mode: Optional[str] = None,
        max_workers: Optional[int] = None  # Number of parallel workers
    ) -> Dict:
        """Enhanced extraction with parallel processing and configurable confidence aggregation.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for extraction
            timeout: Maximum time per patch in seconds
            aggregation_mode: Override instance aggregation mode
            max_workers: Maximum number of parallel workers (defaults to number of CPU cores)
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

            # Default to number of CPU cores if max_workers not specified
            if max_workers is None:
                max_workers = multiprocessing.cpu_count()

            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all patch processing tasks
                future_to_patch = {
                    executor.submit(self._process_patch, patch, prompt, timeout, idx): idx
                    for idx, patch in enumerate(patches)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_patch):
                    idx = future_to_patch[future]
                    try:
                        result = future.result()
                        if result is not None:
                            text, confidence = result
                            results.append((text, confidence))
                            self.logger.debug(f"Patch {idx} completed with confidence {confidence:.4f}")
                    except Exception as e:
                        self.logger.error(f"Patch {idx} failed: {str(e)}")

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