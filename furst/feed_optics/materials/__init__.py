"""
Models and measurements of the reflective coatings.
"""

from ._materials import (
    wavelength_design,
    reflectance_design,
    thickness_aluminum,
    width_interface,
    wavelength_fit_min,
    wavelength_fit_max,
    angle_witness,
    coating_design,
    coating_witness_measured,
    coating_witness_fit,
)

__all__ = [
    "wavelength_design",
    "reflectance_design",
    "thickness_aluminum",
    "width_interface",
    "wavelength_fit_min",
    "wavelength_fit_max",
    "angle_witness",
    "coating_design",
    "coating_witness_measured",
    "coating_witness_fit",
]
