#!/usr/bin/env python3
"""
Production Registration Runner
=============================

This script is designed for production environments where only the
registration computation is needed. It contains NO visualization code
and is suitable for headless servers, automated pipelines, and batch processing.

Usage:
    python run_registration.py hor_reg_sk500.txt
    python run_registration.py --config hor_reg_sk500.txt --save-results results.pkl
"""

import argparse
import sys
import time
from pathlib import Path

# Import only the computation module
from hor_reg_auto_V1 import hor_reg_auto


def main():
    """Main production registration function."""
    parser = argparse.ArgumentParser(
        description="Production stereo IR-Color registration runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_registration.py hor_reg_sk500.txt
  python run_registration.py --config hor_reg_sk500.txt --save-results results.pkl
  python run_registration.py --config hor_reg_sk500.txt --board-size 8 11
        """
    )
    parser.add_argument("config", nargs="?", help="Configuration file path")
    parser.add_argument("--config", dest="config_alt", help="Configuration file path (alternative)")
    parser.add_argument("--board-size", nargs=2, type=int, default=[8, 11], 
                       help="Chessboard size (cols rows), default: 8 11")
    parser.add_argument("--save-results", type=str, help="Save results to pickle file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output messages")
    parser.add_argument("--timing", action="store_true", help="Show timing information")
    
    args = parser.parse_args()
    
    # Handle config file path
    config_path = args.config or args.config_alt
    if not config_path:
        parser.error("Configuration file path is required")
        return 1
    
    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file '{config_path}' not found", file=sys.stderr)
        return 1
    
    try:
        # Start timing if requested
        if args.timing:
            start_time = time.time()
        
        # Run registration
        if not args.quiet:
            print(f"Running registration with config: {config_path}")
            print(f"Board size: {args.board_size[0]}×{args.board_size[1]}")
        
        results = hor_reg_auto(config_path, board_size=tuple(args.board_size))
        
        # Show timing if requested
        if args.timing:
            elapsed_time = time.time() - start_time
            print(f"Registration completed in {elapsed_time:.2f} seconds")
        
        # Print basic results
        if not args.quiet:
            print(f"✅ Registration completed successfully!")
            print(f"📊 Results:")
            print(f"   • Corners detected: {results['error_stats']['total_corners']}")
            print(f"   • RMS X error: {results['error_stats']['rms_x']:.3f} pixels")
            print(f"   • RMS Y error: {results['error_stats']['rms_y']:.3f} pixels")
            print(f"   • Parameters saved to: {results['config']['params_path']}")
        
        # Save results to pickle if requested
        if args.save_results:
            import pickle
            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f)
            if not args.quiet:
                print(f"💾 Results saved to: {args.save_results}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Registration failed: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 