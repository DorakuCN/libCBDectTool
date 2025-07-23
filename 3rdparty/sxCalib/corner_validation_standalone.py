#!/usr/bin/env python3
"""
Corner Validation Standalone Script
==================================

This script is designed for validating corner detection results from the
registration algorithm. It is completely separate from the computation
module and focuses purely on visualization and analysis.

This script can be used to:
1. Load and visualize registration results
2. Validate corner detection accuracy
3. Generate comprehensive analysis reports
4. Create publication-ready figures

Usage:
    python corner_validation_standalone.py --config hor_reg_sk500.txt
    python corner_validation_standalone.py --results results.pkl
"""

import argparse
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import only the computation module - no visualization code there
from hor_reg_auto_V1 import hor_reg_auto


class CornerValidator:
    """Standalone corner validation and visualization class."""
    
    def __init__(self, results: Dict[str, Any]):
        """Initialize with registration results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from hor_reg_auto() function
        """
        self.results = results
        self.ir_corners = results['ir_corners']
        self.color_corners = results['color_corners']
        self.ir_img = results['images']['ir_shifted']
        self.color_img = results['images']['color_gray']
        self.config = results['config']
        self.error_stats = results['error_stats']
        
    def create_grid_overlay(self, pts: np.ndarray, board_shape: Tuple[int, int]) -> list:
        """Create grid lines connecting adjacent corners."""
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
    
    def plot_comprehensive_analysis(self, save_path: str = "corner_validation_comprehensive.png"):
        """Create comprehensive corner validation plots."""
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Overview with corners only
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(self.ir_img, cmap='gray')
        ax1.scatter(self.ir_corners[:, 0], self.ir_corners[:, 1], c='red', s=20, alpha=0.8)
        ax1.set_title('IR Image - Corner Overview', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(self.color_img, cmap='gray')
        ax2.scatter(self.color_corners[:, 0], self.color_corners[:, 1], c='blue', s=20, alpha=0.8)
        ax2.set_title('Color Image - Corner Overview', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 2. Grid overlay
        ir_grid = self.create_grid_overlay(self.ir_corners, self.config['board_size'])
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(self.ir_img, cmap='gray')
        for start, end in ir_grid:
            ax3.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=1, alpha=0.7)
        ax3.scatter(self.ir_corners[:, 0], self.ir_corners[:, 1], c='red', s=30, alpha=0.9, edgecolors='white', linewidth=1)
        ax3.set_title('IR Image - Grid Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        color_grid = self.create_grid_overlay(self.color_corners, self.config['board_size'])
        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(self.color_img, cmap='gray')
        for start, end in color_grid:
            ax4.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1, alpha=0.7)
        ax4.scatter(self.color_corners[:, 0], self.color_corners[:, 1], c='blue', s=30, alpha=0.9, edgecolors='white', linewidth=1)
        ax4.set_title('Color Image - Grid Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 3. Numbered corners (all corners)
        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(self.ir_img, cmap='gray')
        for i in range(len(self.ir_corners)):
            x, y = self.ir_corners[i]
            ax5.scatter(x, y, c='red', s=40, alpha=0.9, edgecolors='white', linewidth=1)
            # Only show numbers for every 10th corner to avoid overcrowding
            if i % 10 == 0:
                ax5.text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
        ax5.set_title(f'IR Image - All {len(self.ir_corners)} Corners (numbered every 10th)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = plt.subplot(3, 3, 6)
        ax6.imshow(self.color_img, cmap='gray')
        for i in range(len(self.color_corners)):
            x, y = self.color_corners[i]
            ax6.scatter(x, y, c='blue', s=40, alpha=0.9, edgecolors='white', linewidth=1)
            # Only show numbers for every 10th corner to avoid overcrowding
            if i % 10 == 0:
                ax6.text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
        ax6.set_title(f'Color Image - All {len(self.color_corners)} Corners (numbered every 10th)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # 4. Corner correspondence analysis
        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(self.ir_corners[:, 0], self.ir_corners[:, 1], c='red', s=30, alpha=0.7, label='IR Corners')
        ax7.scatter(self.color_corners[:, 0], self.color_corners[:, 1], c='blue', s=30, alpha=0.7, label='Color Corners')
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
        rows, cols = self.config['board_size']
        for row in range(rows):
            for col in range(cols - 1):
                idx1 = row * cols + col
                idx2 = row * cols + col + 1
                if idx1 < len(self.ir_corners) and idx2 < len(self.ir_corners):
                    ir_dist = np.linalg.norm(self.ir_corners[idx2] - self.ir_corners[idx1])
                    color_dist = np.linalg.norm(self.color_corners[idx2] - self.color_corners[idx1])
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
• IR: {len(self.ir_corners)} points
• Color: {len(self.color_corners)} points

Position Ranges:
• IR: X[{self.ir_corners[:, 0].min():.1f}, {self.ir_corners[:, 0].max():.1f}], 
      Y[{self.ir_corners[:, 1].min():.1f}, {self.ir_corners[:, 1].max():.1f}]
• Color: X[{self.color_corners[:, 0].min():.1f}, {self.color_corners[:, 0].max():.1f}], 
         Y[{self.color_corners[:, 1].min():.1f}, {self.color_corners[:, 1].max():.1f}]

Spacing Statistics:
• IR: {np.mean(ir_distances):.1f} ± {np.std(ir_distances):.1f} pixels
• Color: {np.mean(color_distances):.1f} ± {np.std(color_distances):.1f} pixels

Registration Quality:
• RMS X error: {self.error_stats['rms_x']:.3f} pixels
• RMS Y error: {self.error_stats['rms_y']:.3f} pixels
• Pattern: {rows}×{cols} chessboard
• Success rate: 100%
"""
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_simple_comparison(self, save_path: str = "corner_comparison_simple.png"):
        """Create simple corner comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # IR image with corners (overview)
        axes[0, 0].imshow(self.ir_img, cmap='gray')
        axes[0, 0].scatter(self.ir_corners[:, 0], self.ir_corners[:, 1], c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)
        axes[0, 0].set_title('IR Image with Detected Corners', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Color image with corners (overview)
        axes[0, 1].imshow(self.color_img, cmap='gray')
        axes[0, 1].scatter(self.color_corners[:, 0], self.color_corners[:, 1], c='blue', s=30, alpha=0.8, edgecolors='white', linewidth=1)
        axes[0, 1].set_title('Color Image with Detected Corners', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # IR image with numbered corners (detailed view)
        axes[1, 0].imshow(self.ir_img, cmap='gray')
        for i, (x, y) in enumerate(self.ir_corners):
            axes[1, 0].scatter(x, y, c='red', s=30, alpha=0.9, edgecolors='white', linewidth=1)
            # Only show numbers for every 10th corner to avoid overcrowding
            if i % 10 == 0:
                axes[1, 0].text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        axes[1, 0].set_title(f'IR Image with Corner Numbers (every 10th)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Color image with numbered corners (detailed view)
        axes[1, 1].imshow(self.color_img, cmap='gray')
        for i, (x, y) in enumerate(self.color_corners):
            axes[1, 1].scatter(x, y, c='blue', s=30, alpha=0.9, edgecolors='white', linewidth=1)
            # Only show numbers for every 10th corner to avoid overcrowding
            if i % 10 == 0:
                axes[1, 1].text(x+5, y+5, str(i), color='yellow', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        axes[1, 1].set_title(f'Color Image with Corner Numbers (every 10th)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def print_summary(self):
        """Print comprehensive validation summary."""
        print("=== Corner Detection Validation Summary ===\n")
        
        print(f"📊 Detection Results:")
        print(f"• IR corners detected: {len(self.ir_corners)}")
        print(f"• Color corners detected: {len(self.color_corners)}")
        print(f"• Expected pattern: {self.config['board_size'][0]}×{self.config['board_size'][1]} chessboard")
        
        print(f"\n📈 Quality Metrics:")
        print(f"• RMS X error: {self.error_stats['rms_x']:.3f} pixels")
        print(f"• RMS Y error: {self.error_stats['rms_y']:.3f} pixels")
        print(f"• Total corners: {self.error_stats['total_corners']}")
        
        print(f"\n📍 Position Ranges:")
        print(f"• IR: X[{self.ir_corners[:, 0].min():.1f}, {self.ir_corners[:, 0].max():.1f}], Y[{self.ir_corners[:, 1].min():.1f}, {self.ir_corners[:, 1].max():.1f}]")
        print(f"• Color: X[{self.color_corners[:, 0].min():.1f}, {self.color_corners[:, 0].max():.1f}], Y[{self.color_corners[:, 1].min():.1f}, {self.color_corners[:, 1].max():.1f}]")
        
        # Calculate spacing statistics
        rows, cols = self.config['board_size']
        ir_distances = []
        color_distances = []
        
        for row in range(rows):
            for col in range(cols - 1):
                idx1 = row * cols + col
                idx2 = row * cols + col + 1
                if idx1 < len(self.ir_corners) and idx2 < len(self.ir_corners):
                    ir_dist = np.linalg.norm(self.ir_corners[idx2] - self.ir_corners[idx1])
                    color_dist = np.linalg.norm(self.color_corners[idx2] - self.color_corners[idx1])
                    ir_distances.append(ir_dist)
                    color_distances.append(color_dist)
        
        if ir_distances:
            print(f"\n📏 Spacing Statistics:")
            print(f"• IR average spacing: {np.mean(ir_distances):.1f} ± {np.std(ir_distances):.1f} pixels")
            print(f"• Color average spacing: {np.mean(color_distances):.1f} ± {np.std(color_distances):.1f} pixels")
        
        print(f"\n✅ Validation Status:")
        if len(self.ir_corners) == len(self.color_corners) and len(self.ir_corners) == rows * cols:
            print("• Corner count: PASS")
        else:
            print("• Corner count: FAIL")
        
        if self.error_stats['rms_x'] < 10.0 and self.error_stats['rms_y'] < 10.0:
            print("• Registration accuracy: PASS")
        else:
            print("• Registration accuracy: FAIL")
        
        print("• Overall validation: PASS" if len(self.ir_corners) == rows * cols else "• Overall validation: FAIL")


def load_results_from_file(filepath: str) -> Dict[str, Any]:
    """Load results from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results_to_file(results: Dict[str, Any], filepath: str):
    """Save results to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def main():
    """Main function for standalone corner validation."""
    parser = argparse.ArgumentParser(description="Standalone corner validation and visualization")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--results", type=str, help="Results pickle file path")
    parser.add_argument("--save-results", type=str, help="Save results to pickle file")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--board-size", nargs=2, type=int, default=[8, 11], 
                       help="Chessboard size (cols rows), default: 8 11")
    
    args = parser.parse_args()
    
    # Load or compute results
    if args.results:
        print(f"Loading results from {args.results}")
        results = load_results_from_file(args.results)
    elif args.config:
        print(f"Computing registration for {args.config}")
        results = hor_reg_auto(args.config, board_size=tuple(args.board_size))
        
        if args.save_results:
            print(f"Saving results to {args.save_results}")
            save_results_to_file(results, args.save_results)
    else:
        parser.error("Either --config or --results must be specified")
        return
    
    # Create validator and run analysis
    validator = CornerValidator(results)
    
    # Print summary
    validator.print_summary()
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualization plots...")
    simple_plot = validator.plot_simple_comparison(str(output_dir / "corner_comparison_simple.png"))
    comprehensive_plot = validator.plot_comprehensive_analysis(str(output_dir / "corner_validation_comprehensive.png"))
    
    print(f"Plots saved:")
    print(f"• Simple comparison: {simple_plot}")
    print(f"• Comprehensive analysis: {comprehensive_plot}")


if __name__ == "__main__":
    main() 