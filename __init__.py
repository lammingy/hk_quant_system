"""
hk_quant_system - Institutional Quantitative Trading & Risk Management Platform for HKEX Equities
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__version__ = "2.0.0"
