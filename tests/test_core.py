import unittest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import torch
import os
import numpy as np
from patchfinder.core import PatchFinder, TransformersPatchFinder, VLLMPatchFinder, MLXPatchFinder

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
        
    @patch('patchfinder.core.mx')
    def test_wrap_mlx_model(self, mock_mx):
        """Test wrap with an MLX model."""
        class MockMLXModel:
            def __repr__(self):
                return "<class 'mlx.model.Model'>"
        
        mlx_model = MockMLXModel()
        finder = PatchFinder.wrap(mlx_model, self.processor)
        self.assertIsInstance(finder, MLXPatchFinder)
        
    @patch('patchfinder.core.mx')
    def test_wrap_mlx_model_no_package(self, mock_mx):
        """Test wrap with MLX model but missing package."""
        mock_mx.side_effect = ImportError()
        
        class MockMLXModel:
            def __repr__(self):
                return "<class 'mlx.model.Model'>"
        
        mlx_model = MockMLXModel()
        with self.assertRaises(ImportError):
            PatchFinder.wrap(mlx_model, self.processor)

class TestMLXPatchFinder(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.processor = Mock()
        self.processor.tokenizer = Mock()
        self.processor.tokenizer.detokenizer = Mock()
        self.processor.tokenizer.eos_token_id = 2
        
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
        """Test MLXPatchFinder initialization"""
        finder = MLXPatchFinder(self.model, self.processor)
        self.assertEqual(finder.patch_size, 256)
        self.assertEqual(finder.overlap, 0.25)
        self.assertEqual(finder.aggregation_mode, "max")  # Test default mode
        
        # Test custom mode
        finder = MLXPatchFinder(self.model, self.processor, aggregation_mode="average")
        self.assertEqual(finder.aggregation_mode, "average")
        
        # Test invalid mode
        with self.assertRaises(ValueError):
            MLXPatchFinder(self.model, self.processor, aggregation_mode="invalid")
            
    def test_aggregation_modes(self):
        """Test different confidence aggregation modes"""
        finder = MLXPatchFinder(self.model, self.processor)
        
        # Mock patches and processing
        patches = [
            Image.new('RGB', (256, 256), color='white'),
            Image.new('RGB', (256, 256), color='black'),
            Image.new('RGB', (256, 256), color='gray')
        ]
        
        # Mock processor
        self.processor.return_value = {
            "input_ids": np.array([[1, 2, 3]])
        }
        
        # Mock different confidences for patches
        confidences = [0.6, 0.9, 0.3]
        texts = ["medium conf", "high conf", "low conf"]
        
        def mock_generate_step(*args, **kwargs):
            for conf in confidences:
                yield 5, np.array([conf, 1 - conf])
        
        self.model.generate_step.side_effect = mock_generate_step
        
        # Mock tokenizer with different texts
        text_iter = iter(texts)
        self.processor.tokenizer.detokenizer.text = lambda: next(text_iter)
        
        with patch('patchfinder.core.generate_patches', return_value=patches):
            # Test max mode (default)
            result = finder.extract(self.test_image)
            self.assertEqual(result["text"], "high conf")
            self.assertAlmostEqual(result["confidence"], 0.9, places=4)
            
            # Test min mode
            result = finder.extract(self.test_image, aggregation_mode="min")
            self.assertEqual(result["text"], "high conf")  # Still highest conf text
            self.assertAlmostEqual(result["confidence"], 0.3, places=4)
            
            # Test average mode
            result = finder.extract(self.test_image, aggregation_mode="average")
            self.assertEqual(result["text"], "high conf")  # Still highest conf text
            self.assertAlmostEqual(result["confidence"], 0.6, places=4)
            
            # Test override at extract time
            finder = MLXPatchFinder(self.model, self.processor, aggregation_mode="min")
            result = finder.extract(self.test_image, aggregation_mode="max")
            self.assertEqual(result["text"], "high conf")
            self.assertAlmostEqual(result["confidence"], 0.9, places=4)
            
    @patch('patchfinder.core.generate_patches')
    def test_extract_basic(self, mock_generate_patches):
        """Test basic extraction with MLX"""
        finder = MLXPatchFinder(self.model, self.processor)
        
        # Mock patch generation
        test_patch = Image.new('RGB', (256, 256), color='white')
        mock_generate_patches.return_value = [test_patch]
        
        # Mock processor
        mock_input_ids = np.array([[1, 2, 3]])
        self.processor.return_value = {
            "input_ids": mock_input_ids
        }
        
        # Mock MLX generation
        mock_token = 5
        mock_logprobs = np.array([0.9, 0.1])
        self.model.generate_step.return_value = [
            (mock_token, mock_logprobs)
        ]
        
        # Mock tokenizer
        self.processor.tokenizer.detokenizer.text = "test text"
        
        # Test extraction
        result = finder.extract(self.test_image)
        self.assertEqual(result["text"], "test text")
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["processed_patches"], 1)
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_multiple_patches(self, mock_generate_patches):
        """Test extraction with multiple patches"""
        finder = MLXPatchFinder(self.model, self.processor)
        
        # Mock patches
        patches = [
            Image.new('RGB', (256, 256), color='white'),
            Image.new('RGB', (256, 256), color='black')
        ]
        mock_generate_patches.return_value = patches
        
        # Mock processor
        self.processor.return_value = {
            "input_ids": np.array([[1, 2, 3]])
        }
        
        # Mock different confidences for patches
        def mock_generate_step(*args, **kwargs):
            for i, confidence in enumerate([0.6, 0.9]):
                yield 5, np.array([confidence, 1 - confidence])
        
        self.model.generate_step.side_effect = mock_generate_step
        
        # Mock tokenizer with different texts
        texts = ["low confidence text", "high confidence text"]
        text_iter = iter(texts)
        self.processor.tokenizer.detokenizer.text = lambda: next(text_iter)
        
        # Test extraction
        result = finder.extract(self.test_image)
        self.assertEqual(result["text"], "high confidence text")
        self.assertGreater(result["confidence"], 0.8)
        self.assertEqual(result["processed_patches"], 2)

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
        mock_output.sequences = torch.tensor([[1, 2, 3, 4]])
        mock_output.scores = [torch.tensor([[[0.9, 0.1]]])]
        self.model.generate.return_value = mock_output
        
        # Mock decoder
        self.processor.decode.return_value = "test text"
        
        # Test extraction
        result = self.patchfinder.extract(self.test_image)
        self.assertEqual(result["text"], "test text")
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["processed_patches"], 1)
        
    @patch('patchfinder.core.generate_patches')
    def test_extract_no_patches(self, mock_generate_patches):
        """Test extraction with no patches"""
        mock_generate_patches.return_value = []
        result = self.patchfinder.extract(self.test_image)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["processed_patches"], 0)

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
        mock_output.outputs = [MagicMock(
            text="test text",
            logprobs=[[0.9, 0.1]]
        )]
        self.llm.generate.return_value = [mock_output]
        
        # Test extraction
        result = finder.extract("test.jpg")
        self.assertEqual(result["text"], "test text")
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["processed_patches"], 1)
