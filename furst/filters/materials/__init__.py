"""
Models and measurements of the visible-blind filter coating.
"""

import astropy.units as u

__all__ = [
    "angle_witness",
    "transmission_design",
    "transmission_witness_measured",
]

# Defined here, above the import of `_materials`, so that Sphinx documents
# it. See `furst.feed_optics.materials` for why.

angle_witness = 0 * u.deg
"""
The angle of incidence at which the witness sample was measured.

The geometry of the measurement was not recorded, so it is taken to be
normal incidence, which is how :cite:t:`ActonSolarBlind` publishes the
transmission of this filter.
The filter is used within a few degrees of normal in any case.
"""

from ._materials import (  # noqa: E402
    transmission_design,
    transmission_witness_measured,
)
