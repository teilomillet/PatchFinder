# confidence.py 
import numpy as np
from scipy.special import softmax
import logging

logger = logging.getLogger(__name__)

def calculate_patch_confidence(logits: np.ndarray) -> float:
    """Robust confidence calculation with detailed diagnostics."""
    try:
        logger.debug(f"Input logits shape: {logits.shape}")
        logger.debug(f"Input logits dtype: {logits.dtype}")
        
        # Log pre-sanitization stats
        logger.debug(f"Pre-sanitization - Min: {np.min(logits):.2f}, Max: {np.max(logits):.2f}, Mean: {np.mean(logits):.2f}, NaN count: {np.isnan(logits).sum()}")
        
        logits = _sanitize_logits(logits)
        
        # Log post-sanitization stats
        logger.debug(f"Post-sanitization - Min: {np.min(logits):.2f}, Max: {np.max(logits):.2f}, Mean: {np.mean(logits):.2f}, NaN count: {np.isnan(logits).sum()}")
        
        probs = softmax(logits, axis=-1)
        logger.debug(f"Softmax output - Min: {np.min(probs):.4f}, Max: {np.max(probs):.4f}, Mean: {np.mean(probs):.4f}")

        max_probs = np.max(probs, axis=-1)
        logger.debug(f"Max probabilities - Min: {np.min(max_probs):.4f}, Max: {np.max(max_probs):.4f}, Mean: {np.mean(max_probs):.4f}")

        confidence = float(np.mean(max_probs))
        logger.info(f"Calculated confidence: {confidence:.4f}")
        return confidence
    
    except Exception as e:
        logger.exception(f"Confidence calculation error. Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
        return 0.0

def _sanitize_logits(logits):
    logits = np.asarray(logits, dtype=np.float32)
    
    if logits.ndim == 3:
        logits = logits[:, -1, :]  # Keep only last token logits
    elif logits.ndim != 2:
        raise ValueError(f"Invalid logits shape: {logits.shape}")
    
    # Handle NaNs/Infs without scaling
    logits = np.nan_to_num(logits, nan=0.0, posinf=1e10, neginf=-1e10)
    return logits  # No division by max_abs