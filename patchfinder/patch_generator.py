# patch_generator.py 
from PIL import Image
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def generate_patches(image_path: str, patch_size: int, overlap: float) -> list:
    """Safe patch generation with validation."""
    try:
        logger.debug(f"Generating patches for {image_path}")
        _validate_inputs(image_path, patch_size, overlap)
        
        # Load and verify image
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # Convert to RGB mode
            img.load()  # Force load the image data
            logger.info(f"Image loaded | Dimensions: {img.size} | Mode: {img.mode}")

            if img.width < patch_size or img.height < patch_size:
                raise ValueError(
                    f"Image too small ({img.width}x{img.height}) "
                    f"for patch size {patch_size}"
                )
            
            patches = _generate_valid_patches(img, patch_size, overlap)
            logger.debug(f"Successfully generated {len(patches)} patches")
            return patches
        finally:
            if 'img' in locals():
                img.close()

    except Exception as e:
        logger.exception("Critical error in patch generation:")
        raise

def _validate_inputs(image_path, patch_size, overlap):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if patch_size <= 0:
        raise ValueError(f"Invalid patch size: {patch_size}")
    if not (0 <= overlap < 1):
        raise ValueError(f"Invalid overlap: {overlap}")

def _generate_valid_patches(img, patch_size, overlap):
    width, height = img.size
    step = max(1, int(patch_size * (1 - overlap)))
    
    logger.debug(
        f"Patch params | Size: {patch_size} | Overlap: {overlap} | Step: {step} | "
        f"Image size: {width}x{height} | Potential patches: "
        f"{(width - patch_size + step) // step * (height - patch_size + step) // step}"
    )

    patches = []
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            try:
                box = (x, y, x + patch_size, y + patch_size)
                patch = img.crop(box)
                
                # Log actual patch content metadata
                logger.debug(f"Patch {len(patches)+1} stats - Mean: {np.mean(patch)}, Std: {np.std(patch)}")
                
                patches.append(patch)
            except Exception as e:
                logger.warning(f"Failed patch at ({x},{y}): {str(e)}")
    
    # Log first patch histogram
    if patches:
        first_patch_array = np.array(patches[0])
        logger.debug(f"First patch histogram: {np.histogram(first_patch_array, bins=5)[0]}")
    
    return patches