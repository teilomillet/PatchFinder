    # confidence.py 
import numpy as np
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)

def calculate_entropy(logits: np.ndarray) -> float:
    """Calculate entropy of token distribution with numerical safeguards.
    
    Args:
        logits: Raw logits or log probabilities [vocab_size]
    Returns:
        Entropy of the distribution
    """
    # Ensure proper log probabilities
    if np.any(logits > 0) or np.max(np.abs(np.exp(logits).sum() - 1)) > 1e-3:
        logits = logits - logsumexp(logits)
    
    # Convert to probabilities with numerical safeguards
    probs = np.exp(logits)
    probs = np.clip(probs, 1e-10, 1.0)  # Prevent log(0)
    probs = probs / probs.sum()  # Renormalize after clipping
    
    return -np.sum(probs * np.log(probs))  # Standard entropy formula

def detect_refusal(text: str) -> float:
    """Returns refusal probability with reduced sensitivity.
    
    Uses a weighted approach where stronger refusal phrases have higher impact.
    Also considers phrase combinations for stronger penalties.
    """
    # Strong refusal phrases (weight: 1.0)
    strong_phrases = {
        "i apologize", "i'm sorry", "cannot extract", "unable to extract",
        "can't read", "cannot read", "unable to read", "don't have access",
        "not clear enough", "too blurry", "image quality"
    }
    
    # Moderate refusal phrases (weight: 0.5)
    moderate_phrases = {
        "sorry", "can't", "unable", "cannot", "difficult to",
        "hard to", "not visible", "not legible", "unclear"
    }
    
    # Context phrases that strengthen refusal when combined (weight: 0.3)
    context_phrases = {
        "image", "text", "document", "content",
        "information", "details", "portion"
    }
    
    text = text.lower()
    
    # Calculate weighted matches
    strong_matches = sum(1.0 for phrase in strong_phrases if phrase in text)
    moderate_matches = sum(0.5 for phrase in moderate_phrases if phrase in text)
    context_matches = sum(0.3 for phrase in context_phrases if phrase in text)
    
    # Combine scores with emphasis on strong refusals
    total_score = strong_matches + moderate_matches
    
    # Add context bonus only if we have some refusal matches
    if total_score > 0:
        total_score += min(context_matches, 1.0)  # Cap context bonus at 1.0
    
    # New normalization and scaling with reduced impact
    return min(1.0, total_score / 5.0) ** 0.5  # Reduced from /3.0 and **0.8

def calculate_patch_confidence(
    logprobs: np.ndarray,
    full_logits: np.ndarray = None,
    vocab_size: int = None,
    min_normalization: bool = True,
    lambda_entropy: float = 1.0,  # Reduced from 1.2
    entropy_floor: float = 0.1,
    use_dynamic_range: bool = False,  # Changed to False to use theoretical bounds
    generated_text: str = None  # Added parameter for refusal detection
) -> float:
    """Implements confidence score using weighted entropy and refusal detection.
    
    Args:
        logprobs: Array of token log probabilities (1D array)
        full_logits: Optional full logits for each token [sequence_length, vocab_size]
        vocab_size: Model's vocabulary size for optional normalization
        min_normalization: Whether to normalize the score to [0,1] range
        lambda_entropy: Weight for entropy penalty (0 disables entropy)
        entropy_floor: Minimum entropy impact (relative to max possible entropy)
        use_dynamic_range: Whether to use observed range for normalization
        generated_text: Optional text for refusal detection
    
    Returns:
        Confidence score between 0 and 1
    """
    try:
        # Convert MLX arrays if needed
        if hasattr(logprobs, '__module__') and 'mlx.core' in logprobs.__module__:
            logprobs = np.array(logprobs)
        if full_logits is not None and hasattr(full_logits, '__module__') and 'mlx.core' in full_logits.__module__:
            full_logits = np.array(full_logits)

        # Validate and preprocess logprobs
        logprobs = validate_and_normalize_logprobs(logprobs)
        
        if len(logprobs) == 0:
            logger.warning("No valid logprobs to calculate confidence")
            return 0.0
        
        # Calculate raw PC score (average log probability per token)
        raw_pc_score = np.mean(logprobs)
        pc_score = raw_pc_score
        
        # Calculate probability statistics for better debugging
        probs = np.exp(logprobs)
        mean_prob = np.mean(probs)
        median_prob = np.median(probs)
        
        logger.debug(
            f"Raw Scores:"
            f"\n  PC Score: {pc_score:.4f}"
            f"\n  Token Count: {len(logprobs)}"
            f"\n  LogProb Range: [{np.min(logprobs):.4f}, {np.max(logprobs):.4f}]"
            f"\n  Prob Range: [{np.exp(np.min(logprobs)):.4f}, {np.exp(np.max(logprobs)):.4f}]"
            f"\n  Mean Prob: {mean_prob:.4f}"
            f"\n  Median Prob: {median_prob:.4f}"
        )
        
        # Calculate entropy penalty if full logits provided
        if full_logits is not None and lambda_entropy > 0:
            # Calculate token-wise entropy ratios
            max_entropy = np.log(vocab_size) if vocab_size else np.log(full_logits.shape[-1])
            entropy_ratios = np.array([
                calculate_entropy(token_logits)/max_entropy 
                for token_logits in full_logits
            ])
            
            # Weight tokens by their probability
            token_weights = np.exp(logprobs)  # Use generated token probabilities
            weighted_entropy = np.sum(entropy_ratios * token_weights) / np.sum(token_weights)
            
            # Apply non-linear penalty with increased sensitivity
            entropy_impact = lambda_entropy * np.tanh(3 * weighted_entropy)
            entropy_impact = np.minimum(entropy_impact, 0.9)  # Cap at 90% reduction
            
            # Apply multiplicative entropy penalty
            raw_score = pc_score
            pc_score = pc_score * (1 - entropy_impact)
            
            logger.debug(
                f"Entropy Analysis:"
                f"\n  Range: [{entropy_ratios.min():.2f}, {entropy_ratios.max():.2f}]"
                f"\n  Weighted Mean: {weighted_entropy:.2f}"
                f"\n  Impact Factor: {entropy_impact:.2f}"
                f"\n  Score Change: {raw_score:.4f} → {pc_score:.4f}"
                f"\n  Reduction: {((raw_score - pc_score) / raw_score * 100):.1f}%"
            )
        
        # Optional normalization
        if min_normalization:
            if use_dynamic_range:
                # Use observed range (not recommended)
                min_logprob = np.percentile(logprobs, 5)
                max_logprob = np.percentile(logprobs, 95)
                logger.debug("Using observed range normalization (not recommended)")
            else:
                # Use tighter theoretical bounds
                if vocab_size:
                    # Tighter bound: assume tokens should be in top 10% of vocabulary
                    min_logprob = np.log(1/(vocab_size * 0.1))
                else:
                    # If no vocab size, use empirical bound based on token probabilities
                    min_logprob = np.log(0.1)  # Expect at least 10% probability
                max_logprob = 0.0  # Perfect prediction per token
                logger.debug("Using tighter theoretical bounds normalization")
            
            # Non-linear normalization with less aggressive squashing
            normalized = (pc_score - min_logprob) / (max_logprob - min_logprob)
            raw_confidence = normalized
            
            # Apply probability-based scaling
            prob_scale = np.clip(mean_prob / 0.5, 0.0, 1.0)  # Scale based on mean probability
            confidence = np.clip(normalized * prob_scale, 0.0, 1.0) ** 0.8  # Less aggressive power
            
            logger.debug(
                f"Normalization:"
                f"\n  Bounds: [{min_logprob:.4f}, {max_logprob:.4f}]"
                f"\n  Raw Normalized: {raw_confidence:.4f}"
                f"\n  Probability Scale: {prob_scale:.4f}"
                f"\n  After Non-linear: {confidence:.4f}"
            )
        else:
            confidence = float(pc_score)
            
        # Apply refusal penalty if text is provided
        if generated_text:
            refusal_penalty = detect_refusal(generated_text)
            if refusal_penalty > 0:
                raw_confidence = confidence
                # Apply stronger base penalty for refusals
                base_penalty = 0.5  # 50% base reduction for any refusal
                confidence = confidence * (1 - refusal_penalty) * base_penalty
                logger.debug(
                    f"Refusal Analysis:"
                    f"\n  Penalty: {refusal_penalty:.2f}"
                    f"\n  Base Penalty: {base_penalty:.2f}"
                    f"\n  Score Impact: {raw_confidence:.4f} → {confidence:.4f}"
                    f"\n  Reduction: {((raw_confidence - confidence) / raw_confidence * 100):.1f}%"
                )
        
        # Apply minimum confidence floor
        confidence = max(0.01, confidence)  # Ensure minimum 1% confidence
        
        logger.debug(f"Final Confidence: {confidence:.4f}")
        return float(confidence)

    except Exception as e:
        logger.error(f"Confidence calculation failed: {str(e)}")
        return 0.0

def validate_and_normalize_logprobs(logprobs: np.ndarray) -> np.ndarray:
    """Ensure valid logprobs format and handle model-specific quirks."""
    try:
        # Handle batch dimension
        if logprobs.ndim == 3:
            logprobs = logprobs.squeeze(0)
        
        # Convert from logits if needed (values > 0 indicate raw logits)
        if np.any(logprobs > 0):
            logger.debug("Converting logits to log probabilities")
            logprobs = logprobs - logsumexp(logprobs)
        
        # Handle NaN/Inf values only
        logprobs = np.nan_to_num(logprobs, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        
        return logprobs
        
    except Exception as e:
        logger.error(f"Logprob validation failed: {str(e)}")
        raise

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