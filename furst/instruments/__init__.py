"""
End-to-end models of the FURST optical system.
"""

from ._instruments import Instrument
from ._design import design

__all__ = [
    "Instrument",
    "design",
]
