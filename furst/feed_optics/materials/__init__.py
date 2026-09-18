"""
Models and measurements of the reflective coatings.
"""

from ._materials import (
    wavelength_design,
    reflectance_design,
    angle_specification,
    thickness_aluminum,
    width_interface,
    coating_design,
    angle_witness,
    coating_witness_measured,
    coating_witness_fit,
)

__all__ = [
    "wavelength_design",
    "reflectance_design",
    "angle_specification",
    "thickness_aluminum",
    "width_interface",
    "coating_design",
    "angle_witness",
    "coating_witness_measured",
    "coating_witness_fit",
]
