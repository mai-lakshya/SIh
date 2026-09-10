"""
Backward-compatibility proxy for backend.api.
Delegates dynamically to backend.07_api.api.
"""
import os
import sys
import importlib

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_BACKEND_DIR)

_SUBDIRS = [
    _WORKSPACE_ROOT,
    _BACKEND_DIR,
    os.path.join(_BACKEND_DIR, "01_intake"),
    os.path.join(_BACKEND_DIR, "02_preprocessing"),
    os.path.join(_BACKEND_DIR, "03_models"),
    os.path.join(_BACKEND_DIR, "04_xai"),
    os.path.join(_BACKEND_DIR, "05_orchestration"),
    os.path.join(_BACKEND_DIR, "06_mlops"),
    os.path.join(_BACKEND_DIR, "07_api"),
    os.path.join(_WORKSPACE_ROOT, "remoteness"),
    os.path.join(_WORKSPACE_ROOT, "frontend"),
]

for _p in _SUBDIRS:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_mod = importlib.import_module("backend.07_api.api")

app = _mod.app

for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)
