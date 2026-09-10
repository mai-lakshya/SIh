"""
NEXUS-XAI Backend Package Initialization.
Ensures sys.path includes backend stages and workspace root for seamless package and standalone script execution.
Exposes core applications and orchestrators.
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

# Dynamically import and expose primary backend entrypoints
try:
    _api_module = importlib.import_module("backend.07_api.api")
    app = _api_module.app
    api = _api_module
except Exception:
    app = None
    api = None

try:
    from risk_analysis_system import RiskAnalysisSystem
except Exception:
    RiskAnalysisSystem = None
