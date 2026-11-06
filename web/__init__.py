"""Web UI components and Streamlit dashboard package."""
try:
    from hk_quant_system.web.components import *
except (ImportError, ValueError):
    from .components import *
