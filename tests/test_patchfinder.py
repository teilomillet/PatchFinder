# test_patchfinder.py
import pytest
from PIL import Image 
from patchfinder import PatchFinder
from transformers import AutoProcessor, AutoModelForCausalLM
import logging
import os
import torch

@pytest.fixture
def vision_model():
    model_name = "microsoft/phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Initialize directly on GPU if available
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return model, processor

def test_full_pipeline(vision_model, capsys):
    model, processor = vision_model
    logger = _configure_test_logger()
    
    # Validate test image
    image_path = os.path.join(os.path.dirname(__file__), "img", "passport3.jpeg")
    with Image.open(image_path) as img:
        logger.info(f"Test image dimensions: {img.size} | Mode: {img.mode}")
        assert img.width >= 256, f"Image width {img.width} < 256"
        assert img.height >= 256, f"Image height {img.height} < 256"

    # Properly initialize PatchFinder with required arguments
    patchfinder = PatchFinder(
        model=model,
        processor=processor,
        patch_size=256,
        overlap=0.25,
    )
    
    # Add actual processing parameters
    result = patchfinder.extract(
        image_path=image_path,
        prompt="Extract all text from this passport document",
        timeout=30
    )
    
    # Detailed output
    with capsys.disabled():
        print("\n=== Processing Report ===")
        print(f"Final Confidence: {result['confidence']:.4f}")
        print(f"Processed Patches: {result.get('processed_patches', 0)}")
        print(f"Image Path: {os.path.abspath(image_path)}")
        if torch.cuda.is_available():
            print(f"GPU Memory Usage: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    # Critical assertions
    assert result["confidence"] > 0, "Zero confidence indicates fundamental processing failure"
    assert result["processed_patches"] > 0, "No patches processed successfully"

def _configure_test_logger():
    logger = logging.getLogger("patchfinder_test")
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler("test_debug.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger