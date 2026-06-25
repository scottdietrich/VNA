"""Launcher script for VNA FMR Measurement System.

Double-click this file or run: python run.py
"""

import os
import sys

# Ensure the package directory is on the import path regardless of
# the working directory (e.g. when double-clicking the file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vna_fmr.app import main

main()
