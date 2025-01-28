# Understanding PatchFinder's Confidence Calculation System

PatchFinder implements a sophisticated confidence scoring system that helps evaluate the reliability of text extracted from images. This document explains how the confidence calculation works, based on the MLX example implementation.

## Overview

The confidence calculation system combines multiple factors to produce a final score between 0 and 1, where:
- 1.0 indicates maximum confidence
- 0.0 indicates minimum confidence

The system takes into account:
1. Token probabilities
2. Entropy of predictions
3. Refusal detection
4. Normalization and scaling

## Configuration

The MLX example demonstrates the key configuration parameters:

```python
# Confidence settings
lambda_entropy: float = 0.8    # Weight for entropy penalty
entropy_floor: float = 0.1     # Minimum entropy impact
min_normalization: bool = True # Whether to normalize scores to [0,1]
use_dynamic_range: bool = True # Use observed range for normalization
```

## Components of Confidence Calculation

### 1. Base Confidence Score

The base confidence score is calculated from the log probabilities of the generated tokens:
- Uses the mean log probability per token
- Higher probabilities indicate more confident predictions
- Raw scores are typically negative (log probabilities)

### 2. Entropy Penalty

The entropy penalty reduces confidence when the model is uncertain:
- Calculated for each token if full logits are available
- Higher entropy indicates more uncertainty
- The impact is weighted by `lambda_entropy` (0.8 in the example)
- Maximum penalty is capped at 90% reduction
- Uses a non-linear (tanh) scaling for better sensitivity

### 3. Refusal Detection

The system detects when the model expresses uncertainty or inability to read the text:

Strong refusal phrases (weight 1.0):
- "i apologize", "i'm sorry"
- "cannot extract", "unable to extract"
- "can't read", "cannot read"
- "not clear enough", "too blurry"

Moderate refusal phrases (weight 0.5):
- "sorry", "can't", "unable"
- "difficult to", "hard to"
- "not visible", "not legible"

Context phrases (weight 0.3):
- "image", "text", "document"
- "information", "details"

The refusal penalty:
- Combines weighted matches
- Applies a base 50% reduction for any refusal
- Scales the penalty based on the strength of refusal

### 4. Normalization

The system offers two normalization approaches:

1. Dynamic Range (configurable):
   - Uses observed min/max log probabilities
   - Takes 5th and 95th percentiles to handle outliers
   - Not recommended for production use

2. Theoretical Bounds (default):
   - Uses vocabulary-based bounds
   - Assumes tokens should be in top 10% of vocabulary
   - More consistent across different inputs

### 5. Final Adjustments

The final confidence score includes:
- Probability-based scaling
- Non-linear normalization
- Minimum confidence floor (1%)

## Aggregation Modes

When processing multiple patches, PatchFinder supports different confidence aggregation modes:

1. `max` (default):
   - Uses highest confidence across patches
   - Best for finding the most reliable patch

2. `min`:
   - Uses lowest confidence across patches
   - Conservative estimate of overall quality

3. `average`:
   - Uses mean confidence across patches
   - Balanced measure of overall quality

## Example Usage

```python
# Initialize PatchFinder with confidence settings
finder = PatchFinder.wrap(
    model, 
    processor,
    patch_size=256,
    overlap=0.25,
    max_workers=1
)

# Extract text with confidence
result = finder.extract(
    image_path,
    prompt="Extract all visible text...",
    aggregation_mode="min"  # Conservative confidence
)

# Access results
text = result["text"]
confidence = result["confidence"]
patches = result["processed_patches"]
```

## Debugging

The system includes comprehensive logging for debugging:
- Raw scores and probabilities
- Entropy analysis
- Normalization steps
- Refusal detection
- Final confidence calculation

Enable DEBUG logging to see detailed information:

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## Best Practices

1. Use theoretical bounds (`use_dynamic_range=False`) for more consistent results
2. Consider `min` aggregation mode for conservative estimates
3. Monitor refusal detection for image quality issues
4. Adjust `lambda_entropy` based on your tolerance for uncertainty
5. Enable DEBUG logging during development for detailed insights

## Implementation Details

The confidence calculation is implemented in `patchfinder/confidence.py` with three main components:

1. `calculate_patch_confidence()`: Main confidence calculation
2. `detect_refusal()`: Text-based refusal detection
3. `calculate_entropy()`: Token distribution entropy calculation

The system is designed to be:
- Numerically stable
- Configurable for different use cases
- Transparent through detailed logging
- Robust to different model outputs 