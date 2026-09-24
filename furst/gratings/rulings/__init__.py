"""
Models of the efficiency of the rulings on the diffraction grating.
"""

import astropy.units as u

__all__ = [
    "depths_simulated",
    "angle_simulated",
    "efficiency_simulated",
    "rulings_simulated",
    "serial_numbers_delivered",
    "angles_delivered",
    "efficiency_delivered",
    "rulings_delivered",
]

depths_simulated = [39, 42, 46] * u.nm
"""
The profile depths at which Zeiss simulated the efficiency of the grating.

The design report :cite:p:`Burkhardt2021` gives the efficiency of the
etched, quasi-sinusoidal profile that Zeiss expected to manufacture at
these three depths, and does not settle on one of them.
The delivered gratings came out 38.5 and 43.7 nm deep; see
:func:`efficiency_delivered`.
"""

angle_simulated = 14.25 * u.deg
"""
The angle of incidence at which Zeiss simulated the efficiency of the
grating.

This is the middle of the 10.5 to 18 degree range that the channels of
the instrument span, and the simulation at this one angle is applied to
all of them.
"""

serial_numbers_delivered = ("ID01", "ID07")
"""
The serial numbers of the two gratings Zeiss delivered.

Both were made to the flight specification.
ID01 flew, and ID07 is the spare.
"""

angles_delivered = [10.50, 14.25, 18.00] * u.deg
"""
The angles of incidence at which Zeiss simulated the efficiency of the
delivered gratings.

The final report :cite:p:`Stock2023` gives the efficiency of each grating
at these three angles, which are the angles of incidence of the first,
middle, and last channels of the instrument.
"""

from ._rulings import (  # noqa: E402
    efficiency_simulated,
    rulings_simulated,
    efficiency_delivered,
    rulings_delivered,
)
