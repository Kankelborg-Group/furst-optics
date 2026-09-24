"""
Models and measurements of the diffraction grating.
"""

import astropy.units as u
import named_arrays as na
from . import materials
from . import rulings
from ._gratings import (
    Grating,
)

__all__ = [
    "materials",
    "rulings",
    "Grating",
    "width_clear_delivered",
    "width_mech_delivered",
]

width_clear_delivered = {
    "ID01": na.Cartesian2dVectorArray(181.90 * u.mm, 36.00 * u.mm),
    "ID07": na.Cartesian2dVectorArray(182.01 * u.mm, 34.56 * u.mm),
}
"""
The width and height of the ruled area of each grating Zeiss delivered,
keyed by serial number.

Zeiss measured these under a microscope for its final report
:cite:p:`Stock2023`.
The width is along the dispersion direction and the height along the
grooves.
Above and below the ruled area, each substrate also carries a strip of
600 lines per millimeter which Zeiss used to align the grooves, and
which is not part of the clear aperture.

The ruled area is not quite centered on the substrate: on ID01 its center
is 0.66 mm from the center of the substrate along the dispersion
direction and 0.80 mm along the grooves, and on ID07 0.30 mm and
0.68 mm.
The report does not record which way round the gratings were mounted, so
these offsets are not modeled.
"""

width_mech_delivered = {
    "ID01": na.Cartesian2dVectorArray(189.97 * u.mm, 59.99 * u.mm),
    "ID07": na.Cartesian2dVectorArray(189.96 * u.mm, 59.98 * u.mm),
}
"""
The width and height of the substrate of each grating Zeiss delivered,
keyed by serial number, as measured for its final report
:cite:p:`Stock2023`.
"""
