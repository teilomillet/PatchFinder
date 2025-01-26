#!/usr/bin/env python3
import fire
import logging
from typing import Optional
from pathlib import Path
from .core import PatchFinder
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class PatchFinderCLI:
    """Command line interface for PatchFinder document processing."""
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-3-vision-128k-instruct",
        patch_size: int = 256,
        overlap: float = 0.25,
        device: Optional[str] = None
    ):
        """Initialize PatchFinder CLI with model configuration.
        
        Args:
            model_name: Name of the vision-language model to use
            patch_size: Size of image patches for processing
            overlap: Overlap ratio between patches (0-1)
            device: Device to run model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy loading of model and processor
        self._model = None
        self._processor = None
        self._finder = None
    
    @property
    def finder(self) -> PatchFinder:
        """Lazy initialization of PatchFinder instance."""
        if self._finder is None:
            if self._model is None:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            self._finder = PatchFinder(
                model=self._model,
                processor=self._processor,
                patch_size=self.patch_size,
                overlap=self.overlap
            )
        return self._finder

    def process(
        self,
        image_path: str,
        prompt: str = "Extract all text from this document",
        timeout: int = 30,
        verbose: bool = False
    ) -> dict:
        """Process a single image file.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for text extraction
            timeout: Timeout in seconds
            verbose: Enable verbose logging
            
        Returns:
            Dictionary containing extraction results
        """
        setup_logging(verbose)
        return self.finder.extract(image_path, prompt, timeout)

    def batch_process(
        self,
        input_dir: str,
        output_file: Optional[str] = None,
        prompt: str = "Extract all text from this document",
        timeout: int = 30,
        verbose: bool = False
    ) -> dict:
        """Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_file: Optional JSON file to save results
            prompt: Custom prompt for text extraction
            timeout: Timeout in seconds per image
            verbose: Enable verbose logging
            
        Returns:
            Dictionary mapping filenames to extraction results
        """
        import json
        from PIL import Image
        
        setup_logging(verbose)
        input_path = Path(input_dir)
        results = {}
        
        if not input_path.is_dir():
            raise ValueError(f"Input path {input_dir} is not a directory")
        
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]
        
        for img_file in image_files:
            try:
                result = self.finder.extract(str(img_file), prompt, timeout)
                results[img_file.name] = result
            except Exception as e:
                logging.error(f"Failed to process {img_file}: {e}")
                results[img_file.name] = {"error": str(e)}
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

def main():
    """Entry point for the CLI."""
    fire.Fire(PatchFinderCLI) 