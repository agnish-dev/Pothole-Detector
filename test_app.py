import unittest
import os
from utils import check_model_exists, calculate_severity

class TestPotholeDetector(unittest.TestCase):
    
    def test_model_file_exists(self):
        """Test if the trained YOLOv8 model weights are present."""
        self.assertTrue(
            check_model_exists('best.pt'), 
            "CRITICAL: The trained model file 'best.pt' is missing from the root directory!"
        )

    def test_requirements_exists(self):
        """Test if requirements.txt exists for successful deployment."""
        self.assertTrue(
            os.path.exists('requirements.txt'), 
            "requirements.txt is missing!"
        )
        
    def test_severity_calculation(self):
        """Test the business logic for severity estimation."""
        # Test High Severity (>5%)
        sev_high, color_high = calculate_severity(600, 10000) # 6%
        self.assertEqual(sev_high, "High")
        self.assertEqual(color_high, (255, 0, 0))
        
        # Test Low Severity (<1%)
        sev_low, color_low = calculate_severity(50, 10000) # 0.5%
        self.assertEqual(sev_low, "Low")
        self.assertEqual(color_low, (0, 255, 0))

if __name__ == '__main__':
    unittest.main()
