"""
Models and measurements of the visible-blind filter in front of the sensor.
"""

import astropy.units as u

__all__ = [
    "thickness_measured",
    "materials",
    "Filter",
]

# Defined here, above the imports, so that Sphinx documents it.
# See `furst.feed_optics.materials` for why.

thickness_measured = [2.04, 2.07] * u.mm
"""
The measured thicknesses of the two magnesium fluoride windows that were
coated to make the flight filter and its spare.

They were specified as 2 mm, thinner than the 5 mm the vendor coats by
default, to reduce the focus shift that the dispersion of magnesium fluoride
introduces across the bandpass.
Which of the two flew is not recorded.
"""

from . import materials  # noqa: E402
from ._filters import Filter  # noqa: E402
