#!/usr/bin/env python3
"""
Setup script for MyoAssist modular package.

This script allows you to install the modular package in development mode,
which makes it importable from anywhere in your Python environment.

To install in development mode, run:
pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="myoassist",
    version="0.1.0",
    description="Modular implementation of MyoAssist for neuromuscular reflex controller optimization",
    author="NEUMove Team",
    packages=find_packages(where="workspace"),
    package_dir={"": "workspace"},
    install_requires=[
        "numpy",
        "matplotlib",
        "cma",
    ],
    python_requires=">=3.6",
) 