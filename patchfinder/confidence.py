    # confidence.py 
import numpy as np
from scipy.special import softmax
import logging
import mlx.core as mx  # Add MLX support

logger = logging.getLogger(__name__)

def calculate_patch_confidence(logits: np.ndarray) -> float:
    """Calculate confidence score from logits or log probabilities.
    
    The confidence calculation handles:
    1. Raw logits from transformers models (2D array: sequence_length x vocab_size)
    2. Log probabilities from MLX (1D array: sequence_length)
    
    We use geometric mean of token probabilities for sequence confidence
    to better handle varying sequence lengths.
    """
    try:
        logger.debug(f"Input shape: {logits.shape}")
        logger.debug(f"Input dtype: {logits.dtype}")
        
        # Convert MLX array to numpy if needed
        if hasattr(logits, '__module__') and 'mlx.core' in logits.__module__:
            logits = np.array(logits)
        
        # Handle 1D array of log probabilities (from MLX)
        if logits.ndim == 1:
            logger.debug("Processing 1D array of log probabilities")
            token_logprobs = logits
        else:
            # Handle 2D/3D array of logits (from transformers)
            logger.debug("Processing logits array")
            logits = _sanitize_logits(logits)
            
            # Convert to log probabilities
            if np.all(logits <= 0):  # Already log probabilities
                token_logprobs = logits.max(axis=-1)
            else:
                # Convert to probabilities and get maximum for each token
                probs = softmax(logits, axis=-1)
                token_logprobs = np.log(probs.max(axis=-1))
        
        # Log stats before confidence calculation
        token_probs = np.exp(token_logprobs)
        logger.debug(
            f"Token probabilities - "
            f"Min: {token_probs.min():.4f}, "
            f"Max: {token_probs.max():.4f}, "
            f"Mean: {token_probs.mean():.4f}"
        )
        
        # Calculate sequence confidence using geometric mean
        # This better handles varying sequence lengths
        eps = 1e-10  # Prevent log(0)
        sequence_confidence = np.exp(np.mean(token_logprobs))
        
        logger.debug(f"Final sequence confidence: {sequence_confidence:.4f}")
        
        return float(sequence_confidence)
        
    except Exception as e:
        logger.exception(f"Confidence calculation failed: {str(e)}")
        return 0.0

def _sanitize_logits(logits):
    """Sanitize and normalize logits array."""
    try:
        logits = np.asarray(logits, dtype=np.float32)
        
        # Handle empty or invalid input
        if logits.size == 0:
            raise ValueError("Empty logits array")
            
        # Handle different input formats
        if logits.ndim == 3:
            # For transformer outputs: (batch, seq_len, vocab_size)
            logits = logits.squeeze(0)  # Remove batch dimension
        elif logits.ndim == 2:
            # For transformers logits: (seq_len, vocab_size)
            pass
        else:
            raise ValueError(f"Invalid logits shape for 2D/3D array: {logits.shape}")
        
        # Handle NaNs/Infs
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Detect if input is already log probabilities
        if np.all(logits <= 0) and np.any(np.exp(logits).sum(axis=-1) - 1.0 < 1e-3):
            logger.debug("Input appears to be log probabilities")
            return logits
            
        # Normalize raw logits to prevent overflow
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        
        return logits
        
    except Exception as e:
        logger.error(f"Error sanitizing logits: {str(e)}")
        raise