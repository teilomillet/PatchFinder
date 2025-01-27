    # confidence.py 
import numpy as np
from scipy.special import softmax
import logging

logger = logging.getLogger(__name__)

def calculate_patch_confidence(logits: np.ndarray) -> float:
    """Calculate confidence score from logits or log probabilities.
    
    The confidence calculation needs to handle two cases:
    1. Raw logits from the model (need softmax)
    2. Log probabilities (need exp)
    
    We also need to:
    1. Prevent overflow/underflow in exponential operations
    2. Handle very sharp distributions gracefully
    3. Account for the full probability distribution, not just the max
    """
    try:
        logger.debug(f"Input logits shape: {logits.shape}")
        logger.debug(f"Input logits dtype: {logits.dtype}")
        
        # Log pre-sanitization stats
        logger.debug(f"Pre-sanitization - Min: {np.min(logits):.2f}, Max: {np.max(logits):.2f}, Mean: {np.mean(logits):.2f}")
        
        # Sanitize and normalize logits
        logits = _sanitize_logits(logits)
        
        # Log post-sanitization stats
        logger.debug(f"Post-sanitization - Min: {np.min(logits):.2f}, Max: {np.max(logits):.2f}, Mean: {np.mean(logits):.2f}")
        
        # Convert to probabilities with numerical stability
        if np.all(logits <= 0):  # Input is log probabilities
            logger.debug("Converting log probabilities to probabilities")
            # Shift to prevent underflow, keeping relative probabilities
            logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits_shifted)
            # Renormalize
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
        else:
            logger.debug("Applying softmax to logits")
            probs = softmax(logits, axis=-1)
            
        logger.debug(f"Probability stats - Min: {np.min(probs):.4f}, Max: {np.max(probs):.4f}, Mean: {np.mean(probs):.4f}")
        
        # Calculate confidence using entropy-based metric
        # This considers the full distribution, not just the max probability
        eps = 1e-10  # Prevent log(0)
        entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
        max_entropy = np.log(probs.shape[-1])  # Maximum possible entropy
        confidence = 1 - (entropy / max_entropy)  # Normalize to [0,1]
        
        # Average across tokens
        mean_confidence = float(np.mean(confidence))
        logger.debug(f"Token-wise confidence stats - Min: {np.min(confidence):.4f}, Max: {np.max(confidence):.4f}, Mean: {mean_confidence:.4f}")
        
        return mean_confidence
        
    except Exception as e:
        logger.exception(f"Confidence calculation failed: {str(e)}")
        return 0.0

def _sanitize_logits(logits):
    """Sanitize and normalize logits/logprobs for confidence calculation."""
    try:
        logits = np.asarray(logits, dtype=np.float32)
        
        # Handle empty or invalid input
        if logits.size == 0:
            raise ValueError("Empty logits array")
            
        # Handle different input formats
        if logits.ndim == 3:
            # For transformer outputs: (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]  # Keep only last token logits
        elif logits.ndim == 2:
            # For MLX/direct logprobs: (seq_len, vocab_size)
            pass
        elif logits.ndim == 1:
            # For single token logprobs
            logits = logits.reshape(1, -1)
        else:
            raise ValueError(f"Invalid logits shape: {logits.shape}")
        
        # Handle NaNs/Infs without scaling
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Detect if input is already log probabilities
        max_abs_val = np.max(np.abs(logits))
        if max_abs_val <= 20:  # Typical range for log probabilities
            logger.debug("Input appears to be log probabilities")
            return logits
            
        # Normalize raw logits
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        
        return logits
        
    except Exception as e:
        logger.error(f"Error sanitizing logits: {str(e)}")
        raise