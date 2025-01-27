#!/usr/bin/env python3

import os
import logging
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from patchfinder import PatchFinder
from PIL import Image

def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for more info
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_test_images():
    """Get paths to test passport images."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(os.path.dirname(script_dir), "tests", "img")
    return [
        os.path.join(img_dir, img)
        for img in ["passport-1.jpeg", "passport2.jpg", "passport3.jpeg"]
    ]

def main():
    logger = setup_logging()
    
    # Get test images
    test_images = get_test_images()
    logger.info(f"Found {len(test_images)} test images")
    
    try:
        # Load MLX model and config
        model_path = "mlx-community/pixtral-12b-4bit"
        logger.info(f"Loading {model_path}...")
        model, processor = load(model_path)
        config = load_config(model_path)
        logger.info("Model loaded successfully")
        
        # Initialize PatchFinder with smaller patches and less overlap
        finder = PatchFinder.wrap(
            model, 
            processor,
            model_path=model_path,  # Pass model_path for config loading
            patch_size=256,  # Smaller patches
            overlap=0.25      # Less overlap
        )
        logger.info("PatchFinder initialized")
        
        # Process each test image
        for img_path in test_images:
            logger.info(f"\nProcessing image: {os.path.basename(img_path)}")
            
            # Process document with different confidence aggregation modes
            for mode in ["max", "min", "average"]:
                logger.info(f"\nTesting with {mode} confidence aggregation:")
                
                # Format prompt using mlx-vlm's template
                prompt = "Please analyze this passport image and extract all visible text and information. Include details like name, date of birth, passport number, and any other readable text."
                formatted_prompt = apply_chat_template(
                    processor, 
                    config, 
                    prompt, 
                    num_images=1
                )
                
                # Extract text using PatchFinder
                result = finder.extract(
                    img_path,
                    prompt=formatted_prompt,
                    aggregation_mode=mode
                )
                
                logger.info(f"Extracted Text: {result['text']}")
                logger.info(f"Confidence Score: {result['confidence']:.4f}")
                logger.info(f"Processed Patches: {result['processed_patches']}")
                
                # Also demonstrate direct MLX-VLM usage for comparison
                logger.info("\nDirect MLX-VLM output:")
                # Resize image if too large to avoid memory issues
                with Image.open(img_path) as img:
                    if img.width > 1024 or img.height > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        resized_path = img_path + ".resized.jpg"
                        img.save(resized_path)
                        result = generate(
                            model, 
                            processor, 
                            formatted_prompt, 
                            image=[resized_path], 
                            verbose=False,
                            temperature=0.0  # No randomness for consistent results
                        )
                        os.remove(resized_path)
                    else:
                        result = generate(
                            model, 
                            processor, 
                            formatted_prompt, 
                            image=[img_path], 
                            verbose=False,
                            temperature=0.0
                        )
                # Handle string or GenerationResult return type
                output_text = result.text if hasattr(result, 'text') else str(result)
                logger.info(f"MLX-VLM Output: {output_text}")
                if hasattr(result, 'logprobs') and result.logprobs is not None:
                    logger.debug(f"First token logprobs: {result.logprobs[0]}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 