"""
An idealized raytrace model of the FURST optical system.
"""

from . import typevars
from . import abc
from . import sources
from . import apertures
from . import feed_optics
from . import gratings
from . import sensors
from . import cameras
from . import spectrographs

__all__ = [
    "typevars",
    "abc",
    "sources",
    "apertures",
    "feed_optics",
    "gratings",
    "sensors",
    "cameras",
    "spectrographs",
]
