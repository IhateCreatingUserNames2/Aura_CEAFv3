# MODIFIED FILE: ceaf_core/modules/mcl_engine/__init__.py

"""MCL Engine Module"""

# Export the CeafSelfRepresentation class
from .self_model import CeafSelfRepresentation

# +++ NEW: EXPORT THE MCLEngine CLASS +++
from .mcl_engine import MCLEngine

__all__ = [
    'CeafSelfRepresentation',
    'self_state_analyzer',
    'MCLEngine'  # <-- ADDED
]