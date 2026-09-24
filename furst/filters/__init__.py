"""
Models and measurements of the visible-blind filter in front of the sensor.
"""

import astropy.units as u

__all__ = [
    "thickness_design",
    "thickness_measured",
    "materials",
    "Filter",
]

# Defined here, above the imports, so that Sphinx documents it.
# See `furst.feed_optics.materials` for why.

thickness_design = 2 * u.mm
"""
The thickness to which the magnesium fluoride windows of the filter were
specified, to within 0.2 mm.

This is thinner than the 5 mm the vendor coats by default, to reduce the
focus shift that the dispersion of magnesium fluoride introduces across the
bandpass.
"""

thickness_measured = [2.04, 2.07] * u.mm
"""
The measured thicknesses of the two magnesium fluoride windows that were
coated to make the flight filter and its spare, both within the
tolerance of :data:`thickness_design`.

Which of the two flew is not recorded.
"""

from . import materials  # noqa: E402
from ._filters import Filter  # noqa: E402
