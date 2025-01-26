# PatchFinder

Based on the paper ["PatchFinder: Leveraging Visual Language Models for Accurate Information Retrieval using Model Uncertainty"](https://www.arxiv.org/pdf/2412.02886)

PatchFinder is a Python library for accurate document text extraction using Vision Language Models (VLMs). It works by splitting images into overlapping patches, processing each patch independently, and combining results based on model confidence.

## Features

- Efficient patch-based document processing
- Support for custom VLM models and prompts
- Confidence-based result aggregation
- GPU acceleration support
- Batch processing capabilities
- Comprehensive CLI interface

## Installation

```bash
pip install patchfinder
```

## Quick Start

Process a single document:

```python
from patchfinder import PatchFinder
from transformers import AutoProcessor, AutoModelForCausalLM

# Initialize model and processor
model_name = "microsoft/phi-3-vision-128k-instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Create PatchFinder instance
finder = PatchFinder(model=model, processor=processor)

# Process image
result = finder.extract(
    image_path="document.jpg",
    prompt="Extract all text from this document"
)

print(f"Extracted text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

## Integration with Existing Models

If you're already using transformers or other vision models, PatchFinder can be easily integrated:

```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from patchfinder import PatchFinder

class DocumentProcessor:
    def __init__(self, model_name="microsoft/phi-3-vision-128k-instruct"):
        # Initialize your existing model pipeline
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add PatchFinder on top
        self.patchfinder = PatchFinder(
            model=self.model,
            processor=self.processor,
            patch_size=256,
            overlap=0.25
        )
    
    def process_document(self, image_path: str, custom_prompt: str = None) -> dict:
        # Use your existing preprocessing if needed
        prompt = custom_prompt or "Extract all text from this document"
        
        # Let PatchFinder handle the patch-based processing
        result = self.patchfinder.extract(
            image_path=image_path,
            prompt=prompt,
            timeout=30
        )
        
        # Post-process or format results as needed
        return {
            "text": result["text"],
            "confidence": result["confidence"],
            "processed_patches": result["processed_patches"]
        }

# Usage example
processor = DocumentProcessor()
result = processor.process_document(
    "document.jpg",
    custom_prompt="Extract and structure all text from this document"
)
```

## Command Line Interface

PatchFinder provides a powerful CLI for both single-file and batch processing:

Process a single image:
```bash
python -m patchfinder.cli process image.jpg --prompt="Extract text" --verbose
```

Process all images in a directory:
```bash
python -m patchfinder.cli batch_process ./images/ --output_file=results.json
```

CLI Options:
- `model_name`: Vision language model to use (default: microsoft/phi-3-vision-128k-instruct)
- `patch_size`: Size of image patches (default: 256)
- `overlap`: Overlap ratio between patches (default: 0.25)
- `device`: Processing device (default: auto-detect GPU/CPU)
- `timeout`: Processing timeout in seconds (default: 30)
- `verbose`: Enable detailed logging

## Advanced Usage

Custom model configuration:
```python
finder = PatchFinder(
    model=model,
    processor=processor,
    patch_size=512,  # Larger patches
    overlap=0.5,     # More overlap
    max_workers=2    # Parallel processing
)
```

Batch processing with custom settings:
```python
from pathlib import Path

image_dir = Path("./documents")
for image_path in image_dir.glob("*.jpg"):
    result = finder.extract(
        image_path=str(image_path),
        prompt="Extract and format all text",
        timeout=60
    )
    print(f"Processed {image_path.name}: {result['confidence']:.2f} confidence")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

