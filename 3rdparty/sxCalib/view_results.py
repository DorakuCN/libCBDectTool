#!/usr/bin/env python3
"""
View Results Script
==================

This script provides a quick overview of all generated visualization results
and helps verify the corner detection accuracy.
"""

import os
from pathlib import Path

def main():
    """Display information about all generated results"""
    print("=== sxCalib Results Overview ===\n")
    
    # Check for main output files
    print("📁 Main Output Files:")
    if os.path.exists("regisD2C.txt"):
        print("✅ regisD2C.txt - Registration parameters")
        with open("regisD2C.txt", "r") as f:
            content = f.read().strip()
            print(f"   Parameters: {content}")
    else:
        print("❌ regisD2C.txt - Not found")
    
    print("\n📊 Visualization Files:")
    
    # Check visualization files
    viz_files = [
        ("corner_comparison.png", "Basic corner comparison"),
        ("corner_validation_detailed.png", "Detailed corner validation"),
        ("data/res/IR_annotated_registration.jpg", "IR image with corners"),
        ("data/res/Color_annotated_registration.jpg", "Color image with corners"),
        ("data/res/IR_annotated.jpg", "IR image (original detection)"),
        ("data/res/Color_annotated.jpg", "Color image (original detection)")
    ]
    
    for filepath, description in viz_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ {filepath}")
            print(f"   {description} ({size:.1f} KB)")
        else:
            print(f"❌ {filepath} - {description}")
    
    print("\n🔍 Corner Detection Summary:")
    print("• Expected pattern: 8×11 chessboard (88 corners)")
    print("• Both IR and Color images should have 88 detected corners")
    print("• Corner positions should form a regular grid pattern")
    print("• Spacing should be consistent within each image")
    
    print("\n📋 Next Steps:")
    print("1. Open corner_comparison.png to see basic corner detection")
    print("2. Open corner_validation_detailed.png for comprehensive analysis")
    print("3. Check regisD2C.txt for registration parameters")
    print("4. Verify corner numbering and grid alignment")
    
    print("\n✅ If all files are present and corner detection looks correct,")
    print("   the registration process was successful!")

if __name__ == "__main__":
    main() 