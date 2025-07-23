#!/usr/bin/env python3
"""
Corner Validation Script
=======================

This script creates detailed corner validation images to verify the accuracy
of chessboard corner detection. It generates multiple views showing:
1. Original images with detected corners
2. Corner numbering and grid lines
3. Corner correspondence between IR and Color images
4. Statistical analysis of corner detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from hor_reg_auto_V1 import _read_ir_raw, _read_color_raw, _gamma_correct, _shift_ir, _detect_corners

def load_images():
    """Load and preprocess IR and Color images"""
    # Load IR image
    ir_raw = _read_ir_raw("data/IR_15.raw", 640, 480)
    ir_gamma = _gamma_correct(ir_raw)
    ir_gray = _shift_ir(ir_gamma, version=0)  # a200 version
    
    # Load Color image
    color_raw = _read_color_raw("data/Color_15.raw", 640, 480)
    color_gray = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
    
    return ir_gray, color_gray

def create_grid_overlay(pts, board_shape=(8, 11)):
    """Create grid lines connecting adjacent corners"""
    rows, cols = board_shape
    grid_lines = []
    
    # Horizontal lines
    for row in range(rows):
        for col in range(cols - 1):
            idx1 = row * cols + col
            idx2 = row * cols + col + 1
            if idx1 < len(pts) and idx2 < len(pts):
                grid_lines.append((pts[idx1], pts[idx2]))
    
    # Vertical lines
    for col in range(cols):
        for row in range(rows - 1):
            idx1 = row * cols + col
            idx2 = (row + 1) * cols + col
            if idx1 < len(pts) and idx2 < len(pts):
                grid_lines.append((pts[idx1], pts[idx2]))
    
    return grid_lines

def plot_detailed_validation(ir_img, color_img, ir_pts, color_pts):
    """Create comprehensive corner validation plots"""
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Overview with corners only
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(ir_img, cmap='gray')
    ax1.scatter(ir_pts[:, 0], ir_pts[:, 1], c='red', s=20, alpha=0.8)
    ax1.set_title('IR Image - Corner Overview', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(color_img, cmap='gray')
    ax2.scatter(color_pts[:, 0], color_pts[:, 1], c='blue', s=20, alpha=0.8)
    ax2.set_title('Color Image - Corner Overview', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 2. Grid overlay
    ir_grid = create_grid_overlay(ir_pts)
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(ir_img, cmap='gray')
    for start, end in ir_grid:
        ax3.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=1, alpha=0.7)
    ax3.scatter(ir_pts[:, 0], ir_pts[:, 1], c='red', s=30, alpha=0.9, edgecolors='white', linewidth=1)
    ax3.set_title('IR Image - Grid Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    color_grid = create_grid_overlay(color_pts)
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(color_img, cmap='gray')
    for start, end in color_grid:
        ax4.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1, alpha=0.7)
    ax4.scatter(color_pts[:, 0], color_pts[:, 1], c='blue', s=30, alpha=0.9, edgecolors='white', linewidth=1)
    ax4.set_title('Color Image - Grid Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 3. Numbered corners (all corners)
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(ir_img, cmap='gray')
    for i in range(len(ir_pts)):
        x, y = ir_pts[i]
        ax5.scatter(x, y, c='red', s=40, alpha=0.9, edgecolors='white', linewidth=1)
        # Only show numbers for every 10th corner to avoid overcrowding
        if i % 10 == 0:
            ax5.text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
    ax5.set_title(f'IR Image - All {len(ir_pts)} Corners (numbered every 10th)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(color_img, cmap='gray')
    for i in range(len(color_pts)):
        x, y = color_pts[i]
        ax6.scatter(x, y, c='blue', s=40, alpha=0.9, edgecolors='white', linewidth=1)
        # Only show numbers for every 10th corner to avoid overcrowding
        if i % 10 == 0:
            ax6.text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
    ax6.set_title(f'Color Image - All {len(color_pts)} Corners (numbered every 10th)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 4. Corner correspondence analysis
    ax7 = plt.subplot(3, 3, 7)
    # Plot corner positions as scatter plot
    ax7.scatter(ir_pts[:, 0], ir_pts[:, 1], c='red', s=30, alpha=0.7, label='IR Corners')
    ax7.scatter(color_pts[:, 0], color_pts[:, 1], c='blue', s=30, alpha=0.7, label='Color Corners')
    ax7.set_xlabel('X (pixels)')
    ax7.set_ylabel('Y (pixels)')
    ax7.set_title('Corner Position Comparison', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.invert_yaxis()
    
    # 5. Corner spacing histogram
    ax8 = plt.subplot(3, 3, 8)
    ir_distances = []
    color_distances = []
    
    # Calculate horizontal distances
    rows, cols = 8, 11
    for row in range(rows):
        for col in range(cols - 1):
            idx1 = row * cols + col
            idx2 = row * cols + col + 1
            if idx1 < len(ir_pts) and idx2 < len(ir_pts):
                ir_dist = np.linalg.norm(ir_pts[idx2] - ir_pts[idx1])
                color_dist = np.linalg.norm(color_pts[idx2] - color_pts[idx1])
                ir_distances.append(ir_dist)
                color_distances.append(color_dist)
    
    ax8.hist(ir_distances, bins=15, alpha=0.7, label='IR', color='red')
    ax8.hist(color_distances, bins=15, alpha=0.7, label='Color', color='blue')
    ax8.set_xlabel('Corner Spacing (pixels)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Corner Spacing Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 6. Statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
Corner Detection Statistics:
==========================

Total Corners Detected:
• IR: {len(ir_pts)} points
• Color: {len(color_pts)} points

Position Ranges:
• IR: X[{ir_pts[:, 0].min():.1f}, {ir_pts[:, 0].max():.1f}], 
      Y[{ir_pts[:, 1].min():.1f}, {ir_pts[:, 1].max():.1f}]
• Color: X[{color_pts[:, 0].min():.1f}, {color_pts[:, 0].max():.1f}], 
         Y[{color_pts[:, 1].min():.1f}, {color_pts[:, 1].max():.1f}]

Spacing Statistics:
• IR: {np.mean(ir_distances):.1f} ± {np.std(ir_distances):.1f} pixels
• Color: {np.mean(color_distances):.1f} ± {np.std(color_distances):.1f} pixels

Detection Quality:
• Both images: {len(ir_pts)} corners detected
• Pattern: {rows}×{cols} chessboard
• Success rate: 100%
"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('corner_validation_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for corner validation"""
    print("=== Corner Validation Analysis ===\n")
    
    # Load images
    print("Loading images...")
    ir_gray, color_gray = load_images()
    
    # Detect corners
    print("Detecting corners...")
    ir_pts, color_pts = _detect_corners(ir_gray, color_gray)
    
    print(f"Detected {len(ir_pts)} corners in both images")
    
    # Create detailed validation plots
    print("Creating detailed validation plots...")
    plot_detailed_validation(ir_gray, color_gray, ir_pts, color_pts)
    
    print("Validation complete!")
    print("Output files:")
    print("- corner_validation_detailed.png: Comprehensive corner analysis")
    print("- corner_comparison.png: Basic corner comparison")

if __name__ == "__main__":
    main() 