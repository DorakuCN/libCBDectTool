#!/usr/bin/env python3
"""
Python binding setup for libcbdetect
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import numpy as np

# 获取numpy头文件路径
numpy_include = np.get_include()

# 定义扩展模块
ext_modules = [
    Extension(
        "libcbdetect",
        sources=[
            "libcbdetect_binding.cpp",
        ],
        include_dirs=[
            numpy_include,
            "../include",
            "../src",
        ],
        libraries=[
            "cbdetect",
            "opencv_core",
            "opencv_imgproc",
            "opencv_imgcodecs",
            "opencv_highgui",
        ],
        library_dirs=[
            "../build",
        ],
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-Wall",
            "-Wextra",
        ],
        extra_link_args=[
            "-std=c++17",
        ],
        language="c++",
    ),
]

# 自定义构建命令
class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # 确保C++库已经编译
        if not os.path.exists("../build/libcbdetect.a"):
            print("Building C++ library...")
            subprocess.check_call(["make"], cwd="../build")
        
        build_ext.build_extension(self, ext)

setup(
    name="libcbdetect",
    version="1.0.0",
    description="Python binding for libcbdetect chessboard detection library",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
    ],
    python_requires=">=3.8",
) 