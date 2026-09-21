"""
Models of the efficiency of the rulings on the diffraction grating.
"""

import astropy.units as u

__all__ = [
    "depths_simulated",
    "angle_simulated",
    "efficiency_simulated",
    "rulings_simulated",
]

depths_simulated = [39, 42, 46] * u.nm
"""
The profile depths at which Zeiss simulated the efficiency of the grating.

The design report :cite:p:`Burkhardt2021` gives the efficiency of the
etched, quasi-sinusoidal profile that Zeiss expected to manufacture at
these three depths, and does not settle on one of them; the depth of the
delivered grating is not recorded.
"""

angle_simulated = 14.25 * u.deg
"""
The angle of incidence at which Zeiss simulated the efficiency of the
grating.

This is the middle of the 10.5 to 18 degree range that the channels of
the instrument span, and the simulation at this one angle is applied to
all of them.
"""

from ._rulings import (  # noqa: E402
    efficiency_simulated,
    rulings_simulated,
)
