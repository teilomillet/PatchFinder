import unittest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import torch
import os
from patchfinder.core import PatchFinder, TransformersPatchFinder, VLLMPatchFinder

class TestPatchFinderWrap(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.processor = Mock()
        
    def test_wrap_transformers_model(self):
        """Test wrap with a Transformers model."""
        finder = PatchFinder.wrap(self.model, self.processor)
        self.assertIsInstance(finder, TransformersPatchFinder)
        
    @patch('patchfinder.core.LLM')
    def test_wrap_vllm_model(self, mock_llm):
        """Test wrap with a vLLM model."""
        vllm_model = Mock(spec=mock_llm)
        finder = PatchFinder.wrap(vllm_model)
        self.assertIsInstance(finder, VLLMPatchFinder)

class TestTransformersPatchFinder(unittest.TestCase):
    def setUp(self):
        # Create mock model and processor
        self.model = Mock()
        self.model.device = 'cpu'
        self.processor = Mock()
        self.patchfinder = TransformersPatchFinder(self.model, self.processor)
        
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
        finder = TransformersPatchFinder(self.model, self.processor)
        self.assertEqual(finder.patch_size, 256)
        self.assertEqual(finder.overlap, 0.25)
        
        # Test custom parameters
        finder = TransformersPatchFinder(self.model, self.processor, patch_size=512, overlap=0.5)
        self.assertEqual(finder.patch_size, 512)
        self.assertEqual(finder.overlap, 0.5)
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            TransformersPatchFinder(self.model, self.processor, patch_size=-1)
        with self.assertRaises(ValueError):
            TransformersPatchFinder(self.model, self.processor, overlap=1.5)
            
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
        mock_output.scores = [torch.tensor([[[0.9, 0.1]]])]
        self.model.generate.return_value = mock_output
        
        # Test extraction
        result = self.patchfinder.extract(self.test_image)
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0)
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_no_patches(self, mock_generate_patches):
        """Test extraction with no patches"""
        mock_generate_patches.return_value = []
        result = self.patchfinder.extract(self.test_image)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)

class TestVLLMPatchFinder(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.llm.sampling_params = Mock(temperature=0, output_logprobs=True)
        
    def test_initialization(self):
        """Test VLLMPatchFinder initialization"""
        finder = VLLMPatchFinder(self.llm)
        self.assertEqual(finder.patch_size, 256)
        self.assertEqual(finder.overlap, 0.25)
        
    def test_invalid_vllm_config(self):
        """Test VLLMPatchFinder with invalid config"""
        self.llm.sampling_params.temperature = 0.7
        with self.assertRaises(ValueError):
            VLLMPatchFinder(self.llm)
            
        self.llm.sampling_params.temperature = 0
        self.llm.sampling_params.output_logprobs = False
        with self.assertRaises(ValueError):
            VLLMPatchFinder(self.llm)
            
    @patch('patchfinder.core.generate_patches')
    def test_extract_basic(self, mock_generate_patches):
        """Test basic extraction with vLLM"""
        finder = VLLMPatchFinder(self.llm)
        
        # Mock patch generation
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Mock vLLM output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(logprobs=[[0.9, 0.1]])]
        self.llm.generate.return_value = [mock_output]
        
        # Test extraction
        result = finder.extract("test.jpg")
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0)
