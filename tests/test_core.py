import unittest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import torch
import os
from patchfinder.core import PatchFinder

class TestPatchFinder(unittest.TestCase):
    def setUp(self):
        # Create mock model and processor
        self.model = Mock()
        self.model.device = 'cpu'
        self.processor = Mock()
        self.patchfinder = PatchFinder(self.model, self.processor)
        
        # Create test directory and images
        self.test_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create standard test image
        self.test_image = os.path.join(self.test_dir, "test_image.png")
        img = Image.new('RGB', (512, 512), color='white')
        img.save(self.test_image)
        
    def tearDown(self):
        # Clean up test images
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
        
    def test_initialization(self):
        """Test initialization with different parameters"""
        # Test default parameters
        finder = PatchFinder(self.model, self.processor)
        self.assertEqual(finder.patch_size, 256)
        self.assertEqual(finder.overlap, 0.25)
        
        # Test custom parameters
        finder = PatchFinder(self.model, self.processor, patch_size=512, overlap=0.5)
        self.assertEqual(finder.patch_size, 512)
        self.assertEqual(finder.overlap, 0.5)
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            PatchFinder(self.model, self.processor, patch_size=-1)
        with self.assertRaises(ValueError):
            PatchFinder(self.model, self.processor, overlap=1.5)
            
    @patch('patchfinder.core.generate_patches')
    def test_extract_basic(self, mock_generate_patches):
        """Test basic extraction functionality"""
        # Mock patch generation
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Mock processor output
        mock_input_ids = torch.tensor([[1, 2, 3]])
        self.processor.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": torch.ones_like(mock_input_ids)
        }
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[[0.9, 0.1]]])
        mock_output.__getitem__.return_value = torch.tensor([1, 2, 3])
        self.model.generate.return_value = mock_output
        
        # Mock processor decode
        self.processor.decode.return_value = "test text"
        
        # Test extraction
        result = self.patchfinder.extract(self.test_image)
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["text"], "test text")
        self.assertGreater(result["confidence"], 0)
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_no_patches(self, mock_generate_patches):
        """Test extraction with no patches"""
        mock_generate_patches.return_value = []
        result = self.patchfinder.extract(self.test_image)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_processor_failure(self, mock_generate_patches):
        """Test handling of processor failures"""
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Simulate processor error
        self.processor.side_effect = RuntimeError("Processor error")
        
        with self.assertRaises(RuntimeError):
            self.patchfinder.extract(self.test_image)
            
    @patch('patchfinder.core.generate_patches')
    def test_extract_model_failure(self, mock_generate_patches):
        """Test handling of model failures"""
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Mock processor output
        mock_input_ids = torch.tensor([[1, 2, 3]])
        self.processor.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": torch.ones_like(mock_input_ids)
        }
        
        # Simulate model error
        self.model.generate.side_effect = RuntimeError("Model error")
        
        with self.assertRaises(RuntimeError):
            self.patchfinder.extract(self.test_image)
            
    @patch('patchfinder.core.generate_patches')
    def test_extract_with_custom_prompt(self, mock_generate_patches):
        """Test extraction with custom prompt"""
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Mock processor and model
        mock_input_ids = torch.tensor([[1, 2, 3]])
        self.processor.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": torch.ones_like(mock_input_ids)
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[[0.9, 0.1]]])
        mock_output.__getitem__.return_value = torch.tensor([1, 2, 3])
        self.model.generate.return_value = mock_output
        
        # Test with custom prompt
        custom_prompt = "Find text in image"
        self.patchfinder.extract(self.test_image, prompt=custom_prompt)
        
        # Verify prompt was passed to processor
        self.processor.assert_called_with(
            images=test_patch,
            text=custom_prompt,
            return_tensors="pt"
        )
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_device_handling(self, mock_generate_patches):
        """Test proper device handling for tensors"""
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Set model device to cuda
        self.model.device = 'cuda'
        
        # Mock processor output with multiple tensors
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_attention_mask = torch.ones_like(mock_input_ids)
        mock_pixel_values = torch.randn(1, 3, 256, 256)
        
        self.processor.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
            "pixel_values": mock_pixel_values,
            "non_tensor_key": "some_value"
        }
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[[0.9, 0.1]]])
        mock_output.__getitem__.return_value = torch.tensor([1, 2, 3])
        self.model.generate.return_value = mock_output
        
        # Extract text
        result = self.patchfinder.extract(self.test_image)
        
        # Verify device handling
        self.model.generate.assert_called_once()
        call_kwargs = self.model.generate.call_args[1]
        
        # Check that tensors were moved to correct device
        self.assertEqual(call_kwargs["input_ids"].device.type, 'cuda')
        self.assertEqual(call_kwargs["attention_mask"].device.type, 'cuda')
        self.assertEqual(call_kwargs["pixel_values"].device.type, 'cuda')
        
        # Check that non-tensor values were preserved
        self.assertEqual(call_kwargs["non_tensor_key"], "some_value")
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_best_confidence(self, mock_generate_patches):
        """Test that the text with highest confidence is selected"""
        # Create multiple patches
        test_patches = [
            Image.new('RGB', (256, 256), color='white'),
            Image.new('RGB', (256, 256), color='black')
        ]
        mock_generate_patches.return_value = test_patches
        
        # Mock processor to return different outputs for different patches
        def processor_side_effect(images, **kwargs):
            if images.getcolors()[0][1] == (255, 255, 255):  # white patch
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            else:  # black patch
                return {"input_ids": torch.tensor([[4, 5, 6]])}
        self.processor.side_effect = processor_side_effect
        
        # Mock model to return different confidences
        def model_side_effect(**kwargs):
            if torch.equal(kwargs["input_ids"], torch.tensor([[1, 2, 3]])):
                output = MagicMock()
                output.logits = torch.tensor([[[0.9, 0.1]]])
                output.__getitem__.return_value = torch.tensor([1, 2, 3])
                return output
            else:
                output = MagicMock()
                output.logits = torch.tensor([[[0.6, 0.4]]])
                output.__getitem__.return_value = torch.tensor([4, 5, 6])
                return output
        self.model.generate.side_effect = model_side_effect
        
        # Mock processor decode to return different texts
        def decode_side_effect(tokens, **kwargs):
            if torch.equal(tokens, torch.tensor([1, 2, 3])):
                return "high confidence text"
            else:
                return "low confidence text"
        self.processor.decode.side_effect = decode_side_effect
        
        # Extract text
        result = self.patchfinder.extract(self.test_image)
        
        # Verify that the text with highest confidence was selected
        self.assertEqual(result["text"], "high confidence text")
        self.assertGreater(result["confidence"], 0.6)  # Lower threshold since we're using softmax
