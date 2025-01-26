# patch_generator.py 
from PIL import Image
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def generate_patches(image_path: str, patch_size: float | int, overlap: float) -> list:
    """Generate patches with dynamic sizing based on image dimensions."""
    try:
        logger.debug(f"Generating patches for {image_path}")
        _validate_inputs(image_path, patch_size, overlap)
        
        # Load image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.load()
        
        # Calculate dynamic patch size
        if isinstance(patch_size, float) and 0 < patch_size < 1:
            min_dim = min(img.width, img.height)
            dynamic_size = int(min_dim * patch_size)
            logger.info(f"Dynamic patch size: {dynamic_size}px")
            patch_size = dynamic_size
        
        # Validate final patch size
        if img.width < patch_size or img.height < patch_size:
            raise ValueError(
                f"Image too small ({img.width}x{img.height}) "
                f"for patch size {patch_size}"
            )
        
        # Generate patches
        patches = _generate_valid_patches(img, patch_size, overlap)
        return patches
        
    finally:
        if 'img' in locals():
            img.close()

def _validate_inputs(image_path, patch_size, overlap):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if isinstance(patch_size, float):
        if not (0 < patch_size <= 1):
            raise ValueError(f"Invalid patch size %: {patch_size}")
    elif isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(f"Invalid patch size: {patch_size}")
    else:
        raise TypeError("patch_size must be int or float")
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