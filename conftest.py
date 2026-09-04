"""
Global pytest configuration.
Imports compatibility shims from compat.py for legacy runners/fixtures without mutating production code.
"""
from compat import apply_all_patches

apply_all_patches()
