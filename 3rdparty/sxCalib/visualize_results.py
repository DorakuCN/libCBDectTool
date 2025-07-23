#!/usr/bin/env python3
"""
Visualization script for stereo IR-Color registration results
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from hor_reg_auto_V1 import _read_ir_raw, _read_color_raw, _gamma_correct, _shift_ir, _detect_corners

def load_and_process_images():
    """Load and process IR and Color images"""
    # Load IR image
    ir_raw = _read_ir_raw("data/IR_15.raw", 640, 480)
    ir_gamma = _gamma_correct(ir_raw)
    ir_gray = _shift_ir(ir_gamma, version=0)  # a200 version
    
    # Load Color image
    color_raw = _read_color_raw("data/Color_15.raw", 640, 480)
    color_gray = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
    
    return ir_gray, color_gray

def detect_and_visualize_corners():
    """Detect corners and create visualization"""
    ir_gray, color_gray = load_and_process_images()
    
    # Detect corners
    ir_pts, color_pts = _detect_corners(ir_gray, color_gray)
    
    # Create visualizations
    ir_vis = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
    color_vis = cv2.cvtColor(color_gray, cv2.COLOR_GRAY2BGR)
    
    # Draw corners (we need to reshape for OpenCV)
    ir_pts_reshaped = ir_pts.reshape(-1, 1, 2)
    color_pts_reshaped = color_pts.reshape(-1, 1, 2)
    
    # Estimate board shape from number of corners (assuming 8x11 pattern)
    board_shape = (8, 11) if len(ir_pts) == 88 else (11, 8)
    
    cv2.drawChessboardCorners(ir_vis, board_shape, ir_pts_reshaped, True)
    cv2.drawChessboardCorners(color_vis, board_shape, color_pts_reshaped, True)
    
    # Add text
    cv2.putText(ir_vis, f"IR: {len(ir_pts)} corners", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(color_vis, f"Color: {len(color_pts)} corners", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return ir_vis, color_vis, ir_pts, color_pts

def plot_corner_comparison(ir_img, color_img, ir_pts, color_pts):
    """Plot corner positions on original images"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # IR image with corners (overview)
    axes[0, 0].imshow(ir_img, cmap='gray')
    axes[0, 0].scatter(ir_pts[:, 0], ir_pts[:, 1], c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)
    axes[0, 0].set_title('IR Image with Detected Corners', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Color image with corners (overview)
    axes[0, 1].imshow(color_img, cmap='gray')
    axes[0, 1].scatter(color_pts[:, 0], color_pts[:, 1], c='blue', s=30, alpha=0.8, edgecolors='white', linewidth=1)
    axes[0, 1].set_title('Color Image with Detected Corners', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # IR image with numbered corners (detailed view)
    axes[1, 0].imshow(ir_img, cmap='gray')
    for i, (x, y) in enumerate(ir_pts):
        axes[1, 0].scatter(x, y, c='red', s=30, alpha=0.9, edgecolors='white', linewidth=1)
        # Only show numbers for every 10th corner to avoid overcrowding
        if i % 10 == 0:
            axes[1, 0].text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    axes[1, 0].set_title(f'IR Image with Corner Numbers (every 10th)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Color image with numbered corners (detailed view)
    axes[1, 1].imshow(color_img, cmap='gray')
    for i, (x, y) in enumerate(color_pts):
        axes[1, 1].scatter(x, y, c='blue', s=30, alpha=0.9, edgecolors='white', linewidth=1)
        # Only show numbers for every 10th corner to avoid overcrowding
        if i % 10 == 0:
            axes[1, 1].text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    axes[1, 1].set_title(f'Color Image with Corner Numbers (every 10th)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('corner_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function"""
    print("=== sxCalib Visualization ===\n")
    
    # Load and process images
    print("Loading and processing images...")
    ir_gray, color_gray = load_and_process_images()
    ir_vis, color_vis, ir_pts, color_pts = detect_and_visualize_corners()
    
    # Save annotated images
    cv2.imwrite("data/res/IR_annotated_registration.jpg", ir_vis)
    cv2.imwrite("data/res/Color_annotated_registration.jpg", color_vis)
    print("Saved annotated images to data/res/")
    
    # Display images
    print("Displaying results...")
    cv2.imshow("IR Image with Corners", ir_vis)
    cv2.imshow("Color Image with Corners", color_vis)
    
    # Create corner comparison plot
    print("Creating corner comparison plot...")
    plot_corner_comparison(ir_gray, color_gray, ir_pts, color_pts)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print statistics
    print(f"\nCorner Detection Statistics:")
    print(f"IR corners: {len(ir_pts)} points")
    print(f"Color corners: {len(color_pts)} points")
    print(f"IR corner range: X[{ir_pts[:, 0].min():.1f}, {ir_pts[:, 0].max():.1f}], Y[{ir_pts[:, 1].min():.1f}, {ir_pts[:, 1].max():.1f}]")
    print(f"Color corner range: X[{color_pts[:, 0].min():.1f}, {color_pts[:, 0].max():.1f}], Y[{color_pts[:, 1].min():.1f}, {color_pts[:, 1].max():.1f}]")
    
    # Calculate and print corner spacing statistics
    if len(ir_pts) > 1:
        ir_distances = []
        color_distances = []
        
        # Calculate distances between adjacent corners (assuming 8x11 pattern)
        rows, cols = 8, 11
        for i in range(len(ir_pts) - 1):
            if i % cols < cols - 1:  # Same row, adjacent columns
                ir_dist = np.linalg.norm(ir_pts[i+1] - ir_pts[i])
                color_dist = np.linalg.norm(color_pts[i+1] - color_pts[i])
                ir_distances.append(ir_dist)
                color_distances.append(color_dist)
        
        if ir_distances:
            print(f"\nCorner Spacing Statistics:")
            print(f"IR average spacing: {np.mean(ir_distances):.2f} ± {np.std(ir_distances):.2f} pixels")
            print(f"Color average spacing: {np.mean(color_distances):.2f} ± {np.std(color_distances):.2f} pixels")
    
    print(f"\nVisualization saved to: corner_comparison.png")
    print(f"Annotated images saved to: data/res/")

if __name__ == "__main__":
    main() 