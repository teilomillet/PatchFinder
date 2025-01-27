# core.py
from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Union, Any, Tuple
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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

class MLXPatchFinder(PatchFinder):
    """PatchFinder implementation for MLX models."""
    
    def __init__(
        self,
        model,
        processor,
        model_path: Optional[str] = None,
        patch_size: Union[int, float] = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        super().__init__(patch_size, overlap, logger, max_workers)
        self.model = model
        self.processor = processor
        self.model_path = model_path
        
        # Load config once during initialization
        try:
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
            
            if self.model_path:
                self.config = load_config(self.model_path)
                self.logger.debug("Successfully loaded MLX-VLM config")
            else:
                self.config = None
                self.logger.warning("No model_path provided for config loading")
        except ImportError as e:
            self.logger.warning(f"MLX-VLM utils not available: {str(e)}")
            self.config = None
            
    def _mlx_generate_with_logprobs(self, prompt_tokens, patch, max_tokens=500):
        """Generate text and compute logprobs using MLX-VLM's generate_step function."""
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs, generate_step
        import numpy as np
        
        self.logger.debug("Starting MLX generation")
        
        try:
            # Prepare inputs for generation
            image_token_index = getattr(self.model.config, "image_token_index", None)
            inputs = prepare_inputs(
                self.processor,
                patch,
                prompt_tokens,
                image_token_index
            )
            
            # Initialize lists for text and logprobs
            generated_text = []
            token_logprobs = []  # Store only the logprob of selected token
            
            # Get eos_token_id from processor or its tokenizer
            eos_token_id = getattr(self.processor, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                self.logger.warning("No eos_token_id found in processor or tokenizer")
                eos_token_id = 2  # Common default, but may need adjustment
            
            # Generate tokens and collect logprobs
            for token, logprobs in generate_step(
                inputs["input_ids"],
                self.model,
                inputs["pixel_values"],
                inputs.get("attention_mask"),
                max_tokens=max_tokens,
                temp=0.0  # Greedy decoding
            ):
                # Convert token to text
                token_text = self.processor.decode([token])
                generated_text.append(token_text)
                
                # Get log probability of the selected token
                # logprobs shape is either [batch_size, vocab_size] or [vocab_size]
                self.logger.debug(f"logprobs shape: {logprobs.shape}, token: {token}")
                
                # Extract the log probability for the selected token
                # Handle both batched and unbatched shapes
                if len(logprobs.shape) == 2:
                    # Remove batch dimension if present (shape: [1, vocab_size])
                    logprobs = logprobs.squeeze(0)  # Now shape: [vocab_size]
                # Now logprobs is always [vocab_size]
                selected_logprob = logprobs[token]
                # Convert to Python scalar
                if isinstance(selected_logprob, mx.array):
                    selected_logprob = selected_logprob.item()
                token_logprobs.append(selected_logprob)
                
                # Debug logging
                self.logger.debug(
                    f"Token: {token_text!r}, "
                    f"LogProb: {selected_logprob:.4f}, "
                    f"Prob: {np.exp(selected_logprob):.4f}"
                )
                
                # Stop if we hit the end token
                if token == eos_token_id:
                    break
            
            # Combine text from all generated tokens
            text = "".join(generated_text)
            
            # Convert token logprobs to numpy array
            if token_logprobs:
                # Shape: (sequence_length,)
                logprobs = np.array(token_logprobs)
                
                # Validation
                probs = np.exp(logprobs)
                self.logger.debug(
                    f"Token probabilities - "
                    f"Min: {probs.min():.4f}, "
                    f"Max: {probs.max():.4f}, "
                    f"Mean: {probs.mean():.4f}"
                )
                self.logger.debug(f"Sequence length: {len(logprobs)}")
            else:
                self.logger.warning("No logprobs generated")
                return None, None
            
            return text, logprobs
            
        except Exception as e:
            self.logger.error(f"MLX generation failed: {str(e)}", exc_info=True)
            return None, None
            
    def _process_patch(self, patch, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # Format prompt using pre-loaded config
            try:
                from mlx_vlm.prompt_utils import apply_chat_template
                
                if self.config is not None:
                    # Use pre-loaded config
                    formatted_prompt = apply_chat_template(
                        self.processor,
                        self.config,
                        prompt,
                        num_images=1
                    )
                else:
                    raise ImportError("No config available")
                    
            except ImportError as e:
                # Fallback to basic template
                self.logger.warning(f"Using basic template: {str(e)}")
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
                formatted_prompt = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # Convert patch if needed
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            
            # Generate with logprobs
            text, logprobs = self._mlx_generate_with_logprobs(
                formatted_prompt,
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