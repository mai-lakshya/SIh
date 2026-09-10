import os
import sys
import importlib.util

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_MLOPS_DIR = os.path.join(_WORKSPACE_ROOT, "backend", "06_mlops")
_TARGET_FILE = os.path.join(_MLOPS_DIR, "scheduler.py")

for _p in [
    _WORKSPACE_ROOT,
    os.path.join(_WORKSPACE_ROOT, "backend"),
    os.path.join(_WORKSPACE_ROOT, "backend", "01_intake"),
    os.path.join(_WORKSPACE_ROOT, "backend", "02_preprocessing"),
    os.path.join(_WORKSPACE_ROOT, "backend", "03_models"),
    os.path.join(_WORKSPACE_ROOT, "backend", "04_xai"),
    os.path.join(_WORKSPACE_ROOT, "backend", "05_orchestration"),
    _MLOPS_DIR,
    os.path.join(_WORKSPACE_ROOT, "backend", "07_api"),
    os.path.join(_WORKSPACE_ROOT, "remoteness"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

spec = importlib.util.spec_from_file_location("_backend_scheduler", _TARGET_FILE)
_mod = importlib.util.module_from_spec(spec)
sys.modules["scheduler"] = _mod
spec.loader.exec_module(_mod)

for attr in dir(_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_mod, attr)
