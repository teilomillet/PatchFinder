"""PatchFinder: A library for accurate document text extraction using Vision Language Models."""

from .core import PatchFinder
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence
from .backends import TransformersPatchFinder, MLXPatchFinder, VLLMPatchFinder

# For backward compatibility
LegacyPatchFinder = TransformersPatchFinder

__version__ = "1.1.0"
__all__ = [
    "PatchFinder",
    "generate_patches",
    "calculate_patch_confidence",
    "TransformersPatchFinder",
    "MLXPatchFinder",
    "VLLMPatchFinder",
    "LegacyPatchFinder"
]