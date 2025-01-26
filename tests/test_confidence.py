import unittest
import numpy as np
from patchfinder.confidence import calculate_patch_confidence

class TestConfidence(unittest.TestCase):
    def test_perfect_confidence(self):
        """Test case where one class has very high probability"""
        logits = np.array([[100.0, 0.0, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertGreater(confidence, 0.999)
        self.assertLess(confidence, 1.001)
        
    def test_low_confidence(self):
        """Test case where all classes have similar probabilities"""
        logits = np.array([[1.0, 1.0, 1.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertAlmostEqual(confidence, 0.333333, places=3)
        
    def test_multiple_tokens(self):
        """Test with multiple tokens"""
        logits = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0]
        ])
        confidence = calculate_patch_confidence(logits)
        self.assertGreater(confidence, 0.999)
        self.assertLess(confidence, 1.001)
        
    def test_empty_input(self):
        """Test with empty input"""
        with self.assertRaises(ValueError):
            calculate_patch_confidence(np.array([]))
            
    def test_inf_values(self):
        # Cas positif
        logits = np.array([[np.inf, 0.0, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertAlmostEqual(confidence, 1.0, places=2)  # Softmax([inf, 0, 0]) → [1, 0, 0]
        
        # Cas négatif
        logits = np.array([[0.0, -np.inf, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertAlmostEqual(confidence, 0.5, places=2)  # Softmax([0, -inf, 0]) → [0.5, 0, 0.5]
        
    def test_nan_values(self):
        """Test with NaN values"""
        logits = np.array([[np.nan, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            calculate_patch_confidence(logits)
            
    def test_large_values(self):
        logits = np.array([[1e30, 0.0, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertAlmostEqual(confidence, 1.0, places=2)
        
    def test_different_shapes(self):
        # 1D → Erreur
        with self.assertRaises(ValueError):
            calculate_patch_confidence(np.array([1.0, 2.0]))
        
        # 3D → Valide après traitement
        logits_3d = np.random.randn(2, 10, 5)  # (batch_size=2, seq_len=10, num_classes=5)
        confidence = calculate_patch_confidence(logits_3d)
        self.assertIsInstance(confidence, float)
            
    def test_zero_probabilities(self):
        """Test with all zero probabilities"""
        logits = np.array([[0.0, 0.0, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertAlmostEqual(confidence, 0.333333, places=3)
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme value differences"""
        # Very large differences between logits
        logits = np.array([[1000.0, -1000.0, 0.0]])
        confidence = calculate_patch_confidence(logits)
        self.assertGreater(confidence, 0.999)
        
        # Very small differences between logits
        logits = np.array([[1e-10, 0.0, -1e-10]])
        confidence = calculate_patch_confidence(logits)
        self.assertGreater(confidence, 0.33)
        self.assertLess(confidence, 0.34)
