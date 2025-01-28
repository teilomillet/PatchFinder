"""MLX backend for PatchFinder."""

import logging
from typing import Optional, Tuple
import numpy as np
from scipy.special import logsumexp
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
            # from mlx_vlm.prompt_utils import apply_chat_template
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
            
    def _get_vocab_size(self) -> Optional[int]:
        """Extract vocabulary size from model configuration or logprobs shape."""
        try:
            # Try to get from model config first
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

    def _mlx_generate_with_logprobs(self, prompt_tokens, patch, max_tokens=42, temperature=0.7, top_p=0.9):
        """Generate text and compute logprobs using MLX-VLM's generate_step function."""
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs, generate_step
        
        self.logger.debug("Starting MLX generation")
        self.logger.debug(f"Generation settings: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
        
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
            full_logits_list = []  # Store full logits distribution
            
            # Get special token IDs
            eos_token_id = getattr(self.processor, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                self.logger.warning("No eos_token_id found in processor or tokenizer")
                eos_token_id = 2  # Common default, but may need adjustment
                
            # Get pad token ID if available
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            
            # Generate tokens and collect logprobs
            for token, logprobs in generate_step(
                inputs["input_ids"],
                self.model,
                inputs["pixel_values"],
                inputs.get("attention_mask"),
                max_tokens=max_tokens,
                temp=temperature,  # Use temperature parameter
                top_p=top_p  # Use top-p parameter
            ):
                # Skip padding tokens
                if pad_token_id is not None and token == pad_token_id:
                    continue
                    
                # Convert token to text
                token_text = self.processor.decode([token])
                
                # Skip empty or whitespace-only tokens
                if not token_text.strip():
                    continue
                
                # Get log probability of the selected token
                # logprobs shape is either [batch_size, vocab_size] or [vocab_size]
                if len(logprobs.shape) == 2:
                    # Remove batch dimension if present (shape: [1, vocab_size])
                    logprobs = logprobs.squeeze(0)  # Now shape: [vocab_size]
                
                # Convert to numpy for processing
                if isinstance(logprobs, mx.array):
                    logprobs = np.array(logprobs)
                
                # Store full logits distribution before normalization
                full_logits_list.append(logprobs.copy())
                
                # Normalize logprobs if they're not already normalized
                if np.any(logprobs > 0):  # Raw logits
                    logprobs = logprobs - logsumexp(logprobs)
                
                # Get the selected token's logprob
                selected_logprob = logprobs[token]
                
                # Include all tokens in the sequence as per paper's formula
                token_logprobs.append(selected_logprob)
                generated_text.append(token_text)
                
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
            
            # Convert token logprobs and full logits to numpy arrays
            if token_logprobs:
                # Shape: (sequence_length,)
                logprobs = np.array(token_logprobs)
                # Shape: (sequence_length, vocab_size)
                full_logits = np.array(full_logits_list)
                
                # Validation
                probs = np.exp(logprobs)
                self.logger.debug(
                    f"Token probabilities - "
                    f"Min: {probs.min():.4f}, "
                    f"Max: {probs.max():.4f}, "
                    f"Mean: {probs.mean():.4f}, "
                    f"Count: {len(logprobs)}"
                )
            else:
                self.logger.warning("No logprobs generated")
                return None, None, None
            
            return text, logprobs, full_logits
            
        except Exception as e:
            self.logger.error(f"MLX generation failed: {str(e)}", exc_info=True)
            return None, None, None

    def _process_patch(self, patch: Image, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"\n{'='*50}")
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
            text, logprobs, full_logits = self._mlx_generate_with_logprobs(
                formatted_prompt,
                patch,
                max_tokens=42,
                temperature=0.3,  # Reduced temperature for more focused sampling
                top_p=0.95  # Slightly increased top_p for better coverage
            )
            
            if text is None or logprobs is None or full_logits is None:
                return None
                
            # Get vocabulary size for proper normalization
            vocab_size = self._get_vocab_size()
            if vocab_size:
                self.logger.debug(f"Using vocabulary size: {vocab_size}")
            
            # Calculate confidence with all parameters including refusal detection
            confidence = calculate_patch_confidence(
                logprobs=logprobs,
                full_logits=full_logits,
                vocab_size=vocab_size,
                lambda_entropy=1.2,
                entropy_floor=0.1,
                use_dynamic_range=False,  # Use theoretical bounds
                generated_text=text  # Pass text for refusal detection
            )
            
            # Debug logging of complete text and stats
            if text:
                self.logger.debug("\nGenerated Text:")
                # Split into sentences and log each one
                sentences = text.split('.')
                for i, sentence in enumerate(sentences, 1):
                    sentence = sentence.strip()
                    if sentence:  # Only log non-empty sentences
                        self.logger.debug(f"  Sentence {i}: {sentence}")
                self.logger.debug(f"\nConfidence: {confidence:.4f}")
            else:
                self.logger.debug("No text generated")
                
            self.logger.debug(f"{'='*50}\n")
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None 