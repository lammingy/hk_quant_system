"""
app.py - Root Streamlit entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from hk_quant_system.web.app import *
except (ImportError, ValueError):
    from web.app import *
