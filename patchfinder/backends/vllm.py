"""vLLM backend for PatchFinder."""

import logging
from typing import Optional, Tuple
from PIL import Image
from ..core import PatchFinder
from ..confidence import calculate_patch_confidence

class VLLMPatchFinder(PatchFinder):
    """PatchFinder implementation for vLLM models."""
    
    def __init__(
        self,
        llm,
        patch_size: int = 256,
        overlap: float = 0.25,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 1
    ):
        super().__init__(patch_size, overlap, logger, max_workers)
        self.llm = llm
        
        # Verify vLLM config
        if not getattr(llm.sampling_params, "temperature", 0) == 0:
            raise ValueError("vLLM temperature must be 0 for confidence calculations")
        if not getattr(llm.sampling_params, "output_logprobs", True):
            raise ValueError("vLLM output_logprobs must be enabled")

    def _process_patch(self, patch: Image, prompt: str, timeout: int, idx: int) -> Optional[Tuple[str, float]]:
        try:
            self.logger.debug(f"Processing patch {idx} | Size: {patch.size} | Mode: {patch.mode}")
            
            # vLLM specific processing here
            outputs = self.llm.generate(prompt, images=[patch])
            
            # Extract text and logprobs
            text = outputs[0].outputs[0].text
            logprobs = outputs[0].outputs[0].logprobs
            confidence = calculate_patch_confidence(logprobs)
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Patch {idx} failed: {str(e)}", exc_info=True)
            return None 