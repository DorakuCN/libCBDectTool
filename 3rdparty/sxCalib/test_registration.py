#!/usr/bin/env python3
"""
Test script for stereo IR-Color registration
This script creates a mock depth file and tests the registration pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from hor_reg_auto_V1 import hor_reg_auto, _read_ir_raw, _read_color_raw, _detect_corners

def create_mock_depth_file(width=640, height=480, output_path="data/Depth_15.raw"):
    """Create a mock depth file for testing purposes"""
    # Create a simple depth pattern (distance increases from top to bottom)
    depth_values = np.linspace(500, 2000, height).reshape(-1, 1) * np.ones(width)
    depth_values = depth_values.astype(np.uint16)
    
    # Add some noise and variation
    noise = np.random.normal(0, 50, depth_values.shape).astype(np.int16)
    depth_values = np.clip(depth_values + noise, 100, 3000).astype(np.uint16)
    
    # Save as raw file
    depth_values.tofile(output_path)
    print(f"Created mock depth file: {output_path}")
    return depth_values

def test_corner_detection():
    """Test corner detection on available images"""
    print("Testing corner detection...")
    
    # Test with BMP files
    ir_bmp = cv2.imread("data/IR.bmp", cv2.IMREAD_GRAYSCALE)
    color_bmp = cv2.imread("data/Color.bmp", cv2.IMREAD_GRAYSCALE)
    
    if ir_bmp is not None and color_bmp is not None:
        try:
            ir_pts, color_pts = _detect_corners(ir_bmp, color_bmp)
            print(f"IR corners detected: {ir_pts.shape[0]} points")
            print(f"Color corners detected: {color_pts.shape[0]} points")
            
            return True
        except Exception as e:
            print(f"Corner detection failed: {e}")
            return False
    else:
        print("Could not load BMP files")
        return False

def test_raw_reading():
    """Test raw file reading functionality"""
    print("Testing raw file reading...")
    
    try:
        # Test IR raw file
        ir_raw = _read_ir_raw("data/IR_15.raw", 640, 480)
        print(f"IR raw file loaded: shape {ir_raw.shape}, dtype {ir_raw.dtype}")
        
        # Test Color raw file
        color_raw = _read_color_raw("data/Color_15.raw", 640, 480)
        print(f"Color raw file loaded: shape {color_raw.shape}, dtype {color_raw.dtype}")
        
        return True
    except Exception as e:
        print(f"Raw file reading failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== sxCalib Registration Test ===\n")
    
    # Test 1: Raw file reading
    if not test_raw_reading():
        print("❌ Raw file reading test failed")
        return
    
    # Test 2: Corner detection
    if not test_corner_detection():
        print("❌ Corner detection test failed")
        return
    
    # Test 3: Create mock depth file
    print("\nCreating mock depth file...")
    create_mock_depth_file()
    
    # Test 4: Full registration pipeline
    print("\nTesting full registration pipeline...")
    try:
        hor_reg_auto("hor_reg_sk500.txt")
        print("✅ Registration completed successfully!")
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 