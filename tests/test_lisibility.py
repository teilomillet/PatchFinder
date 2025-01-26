from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import pytest
from patchfinder import PatchFinder

def test_evaluate_readability():
    # Get the absolute path to the test image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "img", "passport-1.jpeg")
    
    # Model configuration
    model_name = "microsoft/phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        attn_implementation="eager"  # Disable flash attention
    )
    
    # Configure PatchFinder
    patchfinder = PatchFinder(model, processor, patch_size=256, overlap=0.25)
    
    # Extract the average score
    result = patchfinder.extract(image_path, prompt="Read all text clearly and accurately.")
    score = result["confidence"]
    
    # Assert that we get a valid confidence score
    assert 0 <= score <= 1, f"Confidence score {score} should be between 0 and 1"
    print(f"Readability score: {score:.2f}")  # For debugging purposes