# PatchFinder

Based on the paper ["PatchFinder: Leveraging Visual Language Models for Accurate Information Retrieval using Model Uncertainty"](https://www.arxiv.org/pdf/2412.02886)

PatchFinder is a Python library for accurate document text extraction using Vision Language Models (VLMs). It works by splitting images into overlapping patches, processing each patch independently, and combining results based on model confidence.

## Features

- **Lightweight Integration**: 3-line integration with existing VLM code
- **Multiple Backend Support**: Works with Transformers, vLLM, and MLX
- **Zero Configuration**: Automatic backend detection and setup
- **Patch-based Processing**: Efficient document handling with overlapping patches
- **Confidence Scoring**: Built-in uncertainty estimation
- **GPU/Metal Acceleration**: Automatic device management for NVIDIA and Apple Silicon
- **Type Safety**: Full type hints and runtime checks

## Installation

```bash
pip install patchfinder

# For MLX support (Apple Silicon)
pip install mlx-lm
```

## Quick Start

### 1. Transformers Backend (3-Line Integration)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from patchfinder import PatchFinder  # 1. Import

# Your existing model initialization
model_name = "microsoft/phi-3-vision-128k-instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

finder = PatchFinder.wrap(model, processor)  # 2. Wrap

# Process document
result = finder.extract("document.jpg", "Extract all text")  # 3. Use
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

### 2. MLX Backend (Apple Silicon)

```python
from mlx_vlm import load
from patchfinder import PatchFinder  # 1. Import

# Your existing MLX setup
model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")

finder = PatchFinder.wrap(model, processor)  # 2. Wrap

# Process document
result = finder.extract("document.jpg", "Extract all text")  # 3. Use
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

### 3. vLLM Backend (NVIDIA GPUs)

```python
from vllm import LLM
from patchfinder import PatchFinder  # 1. Import

# Your existing vLLM setup
llm = LLM(
    model="microsoft/phi-3-vision-128k-instruct",
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.7
)

finder = PatchFinder.wrap(llm)  # 2. Wrap

# Process document
result = finder.extract("document.jpg", "Extract all text")  # 3. Use
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

## Advanced Usage

### 1. Custom Patch Configuration

```python
finder = PatchFinder.wrap(
    model, 
    processor,
    patch_size=512,  # Larger patches
    overlap=0.5      # More overlap
)
```

### 2. Confidence Aggregation

Configure how patch confidences are aggregated:

```python
# Global configuration
finder = PatchFinder.wrap(
    model, 
    processor,
    aggregation_mode="min"  # 'max', 'min', or 'average'
)

# Per-call override
result = finder.extract(
    "doc.jpg",
    aggregation_mode="average"  # Overrides global setting
)
```

| Mode | Behavior | Use Case |
|------|----------|----------|
| `max` | Highest patch confidence (default) | General-purpose quality assessment |
| `min` | Lowest patch confidence | Critical applications requiring high certainty |
| `average` | Mean of patch confidences | Balanced assessment across document |

### 3. Custom Prompting

```python
result = finder.extract(
    "invoice.pdf",
    prompt="Extract the total amount and date from this invoice",
    timeout=60  # Longer timeout for complex documents
)
```

### 4. Logging Configuration

```python
import logging

logger = logging.getLogger("patchfinder")
logger.setLevel(logging.DEBUG)

finder = PatchFinder.wrap(model, processor, logger=logger)
```

## How It Works

1. **Document Splitting**: Images are split into overlapping patches
2. **Parallel Processing**: Each patch is processed independently
3. **Confidence Scoring**: Model uncertainty is used to score each patch
4. **Result Selection**: Text from highest-confidence patch is returned, with configurable confidence aggregation

## Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

