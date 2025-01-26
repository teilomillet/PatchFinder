import unittest
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from patchfinder.patch_generator import generate_patches

class TestPatchGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create standard test image
        self.standard_image_path = os.path.join(self.test_dir, "test_image.png")
        img = Image.new('RGB', (512, 512), color='white')
        img.save(self.standard_image_path)
        
        # Create small test image
        self.small_image_path = os.path.join(self.test_dir, "small_image.png")
        img_small = Image.new('RGB', (100, 100), color='white')
        img_small.save(self.small_image_path)
        
        # Create JPEG image
        self.jpeg_image_path = os.path.join(self.test_dir, "test_image.jpg")
        img.save(self.jpeg_image_path, 'JPEG')
        
    def tearDown(self):
        # Clean up test images
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
            
    def test_standard_patch_generation(self):
        """Test normal case with default parameters"""
        patch_size = 256
        overlap = 0.25
        patches = generate_patches(self.standard_image_path, patch_size, overlap)
        
        step = int(patch_size * (1 - overlap))
        expected_patches = ((512 - patch_size) // step + 1) ** 2
        
        self.assertEqual(len(patches), expected_patches)
        self.assertEqual(patches[0].size, (patch_size, patch_size))
        
    def test_small_image(self):
        """Test with image smaller than patch size"""
        with self.assertRaises(ValueError):
            generate_patches(self.small_image_path, 256, 0.25)
            
    def test_invalid_image_path(self):
        """Test with non-existent image"""
        with self.assertRaises((FileNotFoundError, UnidentifiedImageError)):
            generate_patches("nonexistent_image.png", 256, 0.25)
            
    def test_different_formats(self):
        """Test with different image formats"""
        png_patches = generate_patches(self.standard_image_path, 256, 0.25)
        jpg_patches = generate_patches(self.jpeg_image_path, 256, 0.25)
        
        self.assertEqual(len(png_patches), len(jpg_patches))
        self.assertEqual(png_patches[0].size, jpg_patches[0].size)
        
    def test_overlap_edge_cases(self):
        """Test edge cases for overlap parameter"""
        # Test no overlap
        patches_no_overlap = generate_patches(self.standard_image_path, 256, 0)
        step = 256
        expected_patches = ((512 - 256) // step + 1) ** 2
        self.assertEqual(len(patches_no_overlap), expected_patches)
        
        # Test maximum overlap
        with self.assertRaises(ValueError):
            generate_patches(self.standard_image_path, 256, 1.0)
            
        # Test negative overlap
        with self.assertRaises(ValueError):
            generate_patches(self.standard_image_path, 256, -0.5)
            
    def test_patch_sizes(self):
        """Test with different patch sizes"""
        # Test very small patches
        small_patches = generate_patches(self.standard_image_path, 32, 0.25)
        self.assertEqual(small_patches[0].size, (32, 32))
        
        # Test large patches
        large_patches = generate_patches(self.standard_image_path, 400, 0.25)
        self.assertEqual(large_patches[0].size, (400, 400))
        
        # Test invalid patch size
        with self.assertRaises(ValueError):
            generate_patches(self.standard_image_path, 0, 0.25)
        with self.assertRaises(ValueError):
            generate_patches(self.standard_image_path, -100, 0.25)
            
    def test_memory_usage(self):
        """Test memory usage with large images"""
        # Create a large test image (4000x4000)
        large_image_path = os.path.join(self.test_dir, "large_image.png")
        large_img = Image.new('RGB', (4000, 4000), color='white')
        large_img.save(large_image_path)
        
        try:
            patches = generate_patches(large_image_path, 256, 0.25)
            # Verify we can access all patches without memory issues
            self.assertTrue(all(isinstance(p, Image.Image) for p in patches))
            self.assertTrue(len(patches) > 0)
        except MemoryError:
            self.fail("Memory error occurred while processing large image")
