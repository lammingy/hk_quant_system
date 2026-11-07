"""
app.py - Root Streamlit entry point.
Executes web/app.py directly on every Streamlit script rerun so clicking buttons,
changing tickers, or switching tabs never hits Python's sys.modules import cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_WEB_APP_FILE = _ROOT / "web" / "app.py"
with open(_WEB_APP_FILE, "rb") as _f:
    _code = compile(_f.read(), str(_WEB_APP_FILE), "exec")
exec(_code, globals())
