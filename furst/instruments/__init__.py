"""
End-to-end models of the FURST optical system.
"""

import astropy.units as u

__all__ = [
    "translation_focus",
    "angle_focus",
    "Instrument",
    "design_proposed",
    "design",
]

# These constants are defined here, rather than beside the functions that
# use them, so that Sphinx documents them, and above the imports which read
# them back out of this partially initialized package.
# See `furst.feed_optics.materials` for why.

translation_focus = 0.8273 * u.mm
"""
The displacement of the feed optic array along the axis of the instrument
which focuses the first channel, found by :meth:`Instrument.focused`.

It is close to the focus shift of the visible-blind filter, since the
instrument images the virtual image of the Sun onto the detector at very
nearly unit magnification, so the array must move away from the grating by
about as much as the filter moves the focus away from it.
"""

angle_focus = -0.00678 * u.deg
"""
The rotation of the feed optic array about the first feed optic which
focuses the last channel, found by :meth:`Instrument.focused`.

This is a very small adjustment, because sliding the array along the axis
of the instrument already focuses the whole spectrum to within a few
microns: the channels sit at increasing azimuth on the Rowland circle, so
that motion carries the long-wavelength channels a little less far in the
radial direction than the short-wavelength ones, which is what the
dispersion of the filter asks for.
The upper stage was a much larger adjustment on the bench, where it also
had to absorb the feed optic array having been re-placed for a grating
whose radius differed from the design.
"""

from ._instruments import Instrument  # noqa: E402
from ._design import design_proposed, design  # noqa: E402
