# __init__.py
from .core import PatchFinder
from .patch_generator import generate_patches
from .confidence import calculate_patch_confidence

__version__ = "1.1.0"
__all__ = ["PatchFinder", "generate_patches", "calculate_patch_confidence"]