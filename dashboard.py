"""
Root entrypoint proxy forwarding to frontend/dashboard.py.
Maintains backward compatibility with streamlit run dashboard.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_ROOT, "backend")
_FRONTEND = os.path.join(_ROOT, "frontend")
for _p in [_ROOT, _BACKEND, _FRONTEND]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FRONTEND_DASHBOARD = os.path.join(_FRONTEND, "dashboard.py")

import runpy
runpy.run_path(_FRONTEND_DASHBOARD, run_name="__main__")
