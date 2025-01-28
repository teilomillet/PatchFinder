#!/usr/bin/env python3

import os
import logging
import time
from dataclasses import dataclass
from typing import List, Dict
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from patchfinder import PatchFinder
from PIL import Image

@dataclass
class Config:
    """Configuration for MLX example."""
    # Model settings
    model_path: str = "mlx-community/pixtral-12b-4bit"
    max_tokens: int = 42
    temperature: float = 0.7  # Temperature for sampling
    top_p: float = 0.9  # Top-p sampling parameter
    
    # PatchFinder settings
    patch_size: int = 256
    overlap: float = 0.25
    max_workers: int = 1
    aggregation_mode: str = "min"
    
    # Confidence settings
    lambda_entropy: float = 0.8  # Increased weight for entropy
    entropy_floor: float = 0.1  # Minimum entropy impact
    min_normalization: bool = True
    use_dynamic_range: bool = True  # Use observed range for normalization
    
    # Prompt settings
    prompt: str = (
        "Extract all visible text and information from that official document. "
        "Don't make any assumptions. Don't invent anything. "
        "Just extract the text and information from the image."
    )

def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.DEBUG,  # Set to INFO for cleaner output
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_test_images():
    """Get paths to test passport images."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(os.path.dirname(script_dir), "tests", "img")
    return [
        os.path.join(img_dir, "usa.png"),
        os.path.join(img_dir, "passport-1.jpeg"),
        os.path.join(img_dir, "civ.png"),
        os.path.join(img_dir, "germany.png"),
        os.path.join(img_dir, "img_canny.png"),
        os.path.join(img_dir, "passport3.jpeg")
    ]

def process_image(
    finder: PatchFinder,
    img_path: str,
    formatted_prompt: str,
    config: Config,
    logger: logging.Logger
) -> Dict:
    """Process a single image and return results."""
    logger.info(f"\nProcessing image: {os.path.basename(img_path)}")
    
    start_time = time.time()
    # Extract text using PatchFinder
    result = finder.extract(
        img_path,
        prompt=formatted_prompt,
        aggregation_mode=config.aggregation_mode
    )
    processing_time = time.time() - start_time
    
    # Calculate time per patch if patches were processed
    time_per_patch = processing_time / result['processed_patches'] if result['processed_patches'] > 0 else 0
    
    return {
        'filename': os.path.basename(img_path),
        'text': result['text'],
        'confidence': result['confidence'],
        'patches': result['processed_patches'],
        'processing_time': processing_time,
        'time_per_patch': time_per_patch
    }

def print_comparison(results: List[Dict], total_time: float, logger: logging.Logger):
    """Print a comparison of results for all images."""
    logger.info("\n" + "="*80)
    logger.info("RESULTS COMPARISON")
    logger.info("="*80)
    
    # Find the longest filename for alignment
    max_filename = max(len(r['filename']) for r in results)
    
    # Print header
    header = f"{'Filename':<{max_filename}} | {'Confidence':^10} | {'Patches':^7} | {'Time (s)':^10} | {'Time/Patch':^12} | Text"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print each result
    total_patches = 0
    for result in results:
        total_patches += result['patches']
        logger.info(
            f"{result['filename']:<{max_filename}} | "
            f"{result['confidence']:^10.4f} | "
            f"{result['patches']:^7d} | "
            f"{result['processing_time']:^10.2f} | "
            f"{result['time_per_patch']:^12.3f} | "
            f"{result['text'][:100]}..."
        )
    
    # Print summary statistics
    logger.info("="*len(header))
    logger.info(f"Total Processing Time: {total_time:.2f} seconds")
    logger.info(f"Average Time per Image: {total_time/len(results):.2f} seconds")
    logger.info(f"Total Patches Processed: {total_patches}")
    logger.info(f"Average Time per Patch: {total_time/total_patches:.3f} seconds")
    logger.info("="*80)

def main():
    logger = setup_logging()
    config = Config()
    
    try:
        # Start timing the entire process
        total_start_time = time.time()
        
        # Load MLX model and config
        logger.info(f"Loading {config.model_path}...")
        model, processor = load(config.model_path)
        mlx_config = load_config(config.model_path)
        logger.info("Model loaded successfully")
        
        # Initialize PatchFinder with all configurable parameters
        finder = PatchFinder.wrap(
            model, 
            processor,
            model_path=config.model_path,
            patch_size=config.patch_size,
            overlap=config.overlap,
            max_workers=config.max_workers
        )
        logger.info("PatchFinder initialized with settings:")
        logger.info(f"  Patch Size: {config.patch_size}")
        logger.info(f"  Overlap: {config.overlap}")
        logger.info(f"  Max Workers: {config.max_workers}")
        logger.info(f"  Aggregation Mode: {config.aggregation_mode}")
        logger.info(f"  Lambda Entropy: {config.lambda_entropy}")
        
        # Format prompt using mlx-vlm's template
        formatted_prompt = apply_chat_template(
            processor, 
            mlx_config, 
            config.prompt, 
            num_images=1
        )
        
        # Process all test images
        results = []
        for img_path in get_test_images():
            result = process_image(finder, img_path, formatted_prompt, config, logger)
            results.append(result)
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        
        # Print comparison of results with timing information
        print_comparison(results, total_time, logger)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 