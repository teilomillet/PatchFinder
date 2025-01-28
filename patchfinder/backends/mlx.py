"""MLX backend for PatchFinder."""

import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from ..core import PatchFinder
from ..confidence import calculate_patch_confidence

class MLXPatchFinder(PatchFinder):
    """PatchFinder implementation for MLX models."""
    
    def __init__(
        self,
        model,
        processor,
        model_path: Optional[str] = None,
        patch_size: int = 256,
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

    def _process_patch(self, patch: Image, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
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