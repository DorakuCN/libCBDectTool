# sxCalib Project Structure

## 📁 **Core Files (Essential)**

### 🔧 **Algorithm Module**
- **`hor_reg_auto_V1.py`** - Main registration algorithm (pure computation)
  - Contains all mathematical computations
  - NO visualization code
  - Returns comprehensive results dictionary
  - Suitable for production environments

### 🚀 **Production Scripts**
- **`run_registration.py`** - Production environment runner
  - Command-line interface for registration
  - Supports batch processing
  - Quiet mode for automation
  - Timing information

- **`corner_validation_standalone.py`** - Standalone validation script
  - Loads and validates registration results
  - Generates comprehensive visualizations
  - Can work with saved results or compute new ones

### 📋 **Configuration & Dependencies**
- **`hor_reg_sk500.txt`** - Configuration file
  - Contains all registration parameters
  - Image paths, camera intrinsics, baselines
  - Hardware version settings

- **`requirements.txt`** - Python dependencies
  - numpy, opencv-python, scipy, matplotlib

- **`sxCheckBoardDetect.py`** - Chessboard detection utilities
  - Corner detection algorithms
  - Multiple scale and channel support
  - Sub-pixel refinement

## 📊 **Data Files**

### 📂 **data/** - Input Data Directory
- **`IR_15.raw`** - Infrared image (16-bit raw format)
- **`Color_15.raw`** - Color image (RGB raw format)
- **`Depth_15.raw`** - Depth image (16-bit raw format)
- **`IR.bmp`** - Infrared image (BMP format)
- **`Color.bmp`** - Color image (BMP format)

### 📂 **data/res/** - Results Directory
- **`IR_annotated.jpg`** - IR image with OpenCV-drawn corners
- **`Color_annotated.jpg`** - Color image with OpenCV-drawn corners
- **`IR_annotated_registration.jpg`** - IR image with registration corners
- **`Color_annotated_registration.jpg`** - Color image with registration corners

## 📈 **Output Files**

### 🔢 **Registration Results**
- **`regisD2C.txt`** - Registration parameters (integer format)
  - 13 polynomial coefficients
  - Hardware-specific formatting

### 📊 **Visualization Results**
- **`corner_comparison_simple.png`** - Basic corner comparison
- **`corner_validation_comprehensive.png`** - Detailed analysis with statistics

## 🔄 **Legacy Files (Deprecated but Functional)**

### 📊 **Mixed Computation/Visualization**
- **`visualize_results.py`** - Basic visualization (legacy)
  - Mixed computation and visualization
  - Still functional but not recommended

- **`corner_validation.py`** - Detailed validation (legacy)
  - Mixed computation and visualization
  - Replaced by standalone version

### 🧪 **Testing & Utilities**
- **`test_registration.py`** - Testing script
  - Creates mock depth files
  - Tests individual components
  - Development and debugging

- **`view_results.py`** - Results overview utility
  - Lists generated files
  - Provides quick status check

## 🛡️ **Configuration Files**

### 🔒 **Version Control**
- **`.gitignore`** - Git ignore rules
  - Excludes Python cache files
  - Excludes temporary and result files
  - Keeps project clean

### 📖 **Documentation**
- **`README.md`** - Main documentation
  - Usage instructions
  - Architecture overview
  - Migration guide

- **`PROJECT_STRUCTURE.md`** - This file
  - Detailed file descriptions
  - Project organization

## 🚀 **Usage Workflow**

### **Production Environment**
```bash
# 1. Run registration (computation only)
python run_registration.py hor_reg_sk500.txt

# 2. Validate results (visualization only)
python corner_validation_standalone.py --results results.pkl
```

### **Development Environment**
```bash
# Test individual components
python test_registration.py

# Quick results overview
python view_results.py
```

## 📋 **File Dependencies**

### **Core Dependencies**
```
hor_reg_auto_V1.py
├── sxCheckBoardDetect.py
├── numpy
├── opencv-python
└── scipy

run_registration.py
└── hor_reg_auto_V1.py

corner_validation_standalone.py
├── hor_reg_auto_V1.py
├── matplotlib
└── pickle
```

### **Data Dependencies**
```
hor_reg_sk500.txt
├── data/IR_15.raw
├── data/Color_15.raw
└── data/Depth_15.raw
```

## 🧹 **Cleanup Rules**

### **Automatically Excluded**
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files
- `*.pkl` - Pickle result files
- `*.npy` - NumPy array files
- `temp/`, `tmp/` - Temporary directories

### **Kept Files**
- `regisD2C.txt` - Final registration parameters
- `corner_comparison_simple.png` - Final visualization
- `corner_validation_comprehensive.png` - Final analysis
- All source code files
- All input data files 