"""
An idealized raytrace model of the FURST optical system.
"""

from ._sources import SolarDisk
from ._front_apertures import FrontAperture

__all__ = [
    "SolarDisk",
    "FrontAperture",
]
