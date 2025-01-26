# document_processor.py
import os
import time
import logging
from PIL import Image
from patchfinder import PatchFinder
from transformers import AutoProcessor, AutoModelForCausalLM

class DocumentProcessor:
    def __init__(self, model_name="microsoft/phi-3-vision-128k-instruct"):
        self.logger = self._configure_logger()
        self.model, self.processor = self._load_model(model_name)
        self.patchfinder = PatchFinder(
            model=self.model,
            processor=self.processor,
            patch_size=256,
            overlap=0.25
        )

    def _configure_logger(self):
        logger = logging.getLogger("DocumentProcessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_model(self, model_name):
        self.logger.info("Loading model and processor...")
        start_time = time.time()
        
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        
        self.logger.info(f"Model loaded in {time.time()-start_time:.2f}s")
        return model, processor

    def process_directory(self, image_dir, prompt="Extract all text from this document"):
        image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        results = []
        for img_path in image_paths:
            try:
                result = self._process_single_image(img_path, prompt)
                results.append((img_path, result))
                self._print_report(img_path, result)
            except Exception as e:
                self.logger.error(f"Failed to process {img_path}: {str(e)}")
                results.append((img_path, None))
        
        return results

    def _process_single_image(self, img_path, prompt):
        with Image.open(img_path) as img:
            if img.width < 256 or img.height < 256:
                raise ValueError(f"Image too small ({img.size}) for processing")
                
        return self.patchfinder.extract(
            image_path=img_path,
            prompt=prompt,
            timeout=60
        )

    def _print_report(self, img_path, result):
        print("\n" + "="*40)
        print(f"Document: {os.path.basename(img_path)}")
        print(f"Status: {'SUCCESS' if result else 'FAILED'}")
        
        if result:
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processed Patches: {result['processed_patches']}")
            print(f"Average Patch Confidence: {result['confidence']:.4f}")
        
        print(f"Full Path: {os.path.abspath(img_path)}")
        print("="*40 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Text Extraction Processor")
    parser.add_argument("--image-dir", default="images", help="Directory containing documents to process")
    parser.add_argument("--prompt", default="Extract all visible text from this document", 
                      help="Processing prompt for the AI model")
    args = parser.parse_args()

    processor = DocumentProcessor()
    processor.process_directory(args.image_dir, args.prompt)