# sxCalib - Stereo IR-Color Registration

This project implements stereo IR-Color registration for depth camera calibration, re-implementing the MATLAB function `hor_reg_auto_V1.m` in Python.

## 🏗️ **Architecture Overview**

The project follows a **separation of concerns** design pattern:

### 🔧 **Computation Module** (Production-Ready)
- **`hor_reg_auto_V1.py`** - Pure computation module, NO visualization
- **`run_registration.py`** - Production runner script
- Suitable for headless servers, automated pipelines, batch processing

### 📊 **Visualization Module** (Validation & Analysis)
- **`corner_validation_standalone.py`** - Standalone validation script
- **`visualize_results.py`** - Basic visualization (legacy)
- **`corner_validation.py`** - Detailed validation (legacy)

## 📁 **Files**

### Core Files
- `hor_reg_auto_V1.py` - Main registration algorithm (pure computation)
- `sxCheckBoardDetect.py` - Chessboard corner detection utilities
- `hor_reg_sk500.txt` - Configuration file for registration parameters
- `requirements.txt` - Python dependencies

### Production Scripts
- `run_registration.py` - Production environment runner
- `corner_validation_standalone.py` - Standalone validation and visualization

### Legacy Scripts (Mixed computation/visualization)
- `visualize_results.py` - Basic visualization (deprecated)
- `corner_validation.py` - Detailed validation (deprecated)
- `test_registration.py` - Testing script
- `view_results.py` - Results overview

## 🚀 **Usage**

### Production Environment (Recommended)

**1. Run Registration (Computation Only)**
```bash
# Basic usage
python run_registration.py hor_reg_sk500.txt

# With custom board size
python run_registration.py --config hor_reg_sk500.txt --board-size 8 11

# Save results for later analysis
python run_registration.py --config hor_reg_sk500.txt --save-results results.pkl --timing

# Quiet mode for automation
python run_registration.py --config hor_reg_sk500.txt --quiet
```

**2. Validate Results (Visualization Only)**
```bash
# Validate from config file (computes + visualizes)
python corner_validation_standalone.py --config hor_reg_sk500.txt

# Validate from saved results (visualization only)
python corner_validation_standalone.py --results results.pkl

# Save results and validate
python corner_validation_standalone.py --config hor_reg_sk500.txt --save-results results.pkl
```

### Legacy Usage (Mixed Computation/Visualization)

```bash
# Install dependencies
pip install -r requirements.txt

# Run registration with basic visualization
python hor_reg_auto_V1.py hor_reg_sk500.txt

# Or use the function directly
python -c "from hor_reg_auto_V1 import hor_reg_auto; hor_reg_auto('hor_reg_sk500.txt')"
```

## 📋 **Configuration File Format**

The configuration file (`hor_reg_sk500.txt`) contains the following parameters:

```
0                    # if_show: 1 to display results, 0 otherwise
0                    # version: 0 for a200, 1 for s300
./data/IR_15.raw     # IR image path
./data/Color_15.raw  # Color image path  
./data/Depth_15.raw  # Depth image path
640                  # Image width
480                  # Image height
2.2                  # fx_rgb: RGB focal length
15                   # color_ir_base: baseline between color and IR
2.5                  # fx_ir: IR focal length
70                   # ir_projector_base: baseline between IR and projector
regisD2C.txt         # Output parameters file
```

## 🔍 **Algorithm Overview**

The registration process:

1. **Image Loading**: Reads IR, Color, and Depth images (raw or standard formats)
2. **Preprocessing**: Applies pixel shifts and gamma correction to IR images
3. **Corner Detection**: Finds chessboard corners in both IR and Color images
4. **Depth Compensation**: Uses depth information to compute disparity compensation
5. **Polynomial Fitting**: Fits 2nd-order polynomial transformation parameters
6. **Error Evaluation**: Computes RMS reprojection errors
7. **Parameter Output**: Saves transformation coefficients to file

## 📊 **Output**

### Computation Results
- **`regisD2C.txt`** - Registration parameters (integer format)
- **Results dictionary** - Comprehensive results including:
  - Corner coordinates (IR and Color)
  - Depth disparity terms
  - Polynomial parameters
  - Error statistics
  - Configuration parameters

### Visualization Output
- **`corner_comparison_simple.png`** - Basic corner comparison
- **`corner_validation_comprehensive.png`** - Detailed analysis with statistics
- **Console output** - Validation summary and quality metrics

## 🏭 **Production Integration**

### Headless Server Usage
```python
from hor_reg_auto_V1 import hor_reg_auto

# Pure computation - no GUI dependencies
results = hor_reg_auto("config.txt")

# Access results programmatically
corners_ir = results['ir_corners']
corners_color = results['color_corners']
rms_error = results['error_stats']['rms_x']
```

### Batch Processing
```bash
# Process multiple configurations
for config in configs/*.txt; do
    python run_registration.py "$config" --save-results "results/$(basename $config .txt).pkl" --quiet
done
```

### Automated Validation
```bash
# Validate all results
for result in results/*.pkl; do
    python corner_validation_standalone.py --results "$result" --output-dir validation/
done
```

## ⚠️ **Missing Files**

**Note**: The `Depth_15.raw` file is currently missing from the data directory. This file is required for the depth-based disparity compensation in the registration algorithm. You'll need to provide this file or modify the script to work without depth information.

## 🔄 **Migration Guide**

### From Legacy to New Architecture

**Old way (mixed computation/visualization):**
```python
from hor_reg_auto_V1 import hor_reg_auto_V1
hor_reg_auto_V1("config.txt")  # Prints results, may show windows
```

**New way (separated concerns):**
```python
# Computation only
from hor_reg_auto_V1 import hor_reg_auto
results = hor_reg_auto("config.txt")  # Returns results dictionary

# Visualization only (if needed)
from corner_validation_standalone import CornerValidator
validator = CornerValidator(results)
validator.print_summary()
validator.plot_comprehensive_analysis()
```

## 📈 **Performance**

- **Computation time**: ~2-5 seconds (depending on image size)
- **Memory usage**: ~50-100 MB (for 640×480 images)
- **Dependencies**: Minimal (numpy, opencv-python, scipy)
- **GUI dependencies**: None (for computation module)

## ✅ **Quality Assurance**

The validation scripts provide comprehensive quality metrics:
- Corner detection accuracy
- Registration precision (RMS errors)
- Grid pattern validation
- Statistical analysis of corner spacing
- Visual verification of results 