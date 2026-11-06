"""Config package for HK Quant System."""
try:
    from hk_quant_system.config.settings import *
except (ImportError, ValueError):
    from .settings import *
