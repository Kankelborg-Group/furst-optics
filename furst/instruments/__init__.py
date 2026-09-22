"""
End-to-end models of the FURST optical system.
"""

import astropy.units as u

__all__ = [
    "wavelength_focus_first",
    "wavelength_focus_last",
    "translation_focus",
    "angle_focus",
    "Instrument",
    "width_line",
    "focus",
    "design_proposed",
    "design",
]

# These constants are defined here, rather than beside the functions that
# use them, so that Sphinx documents them, and above the imports which read
# them back out of this partially initialized package.
# See `furst.feed_optics.materials` for why.

wavelength_focus_first = (600 / 2200) * 434.31 * u.nm
"""
The wavelength at which the first channel of the instrument was focused.

The instrument was focused in visible light, using the coarse alignment
rulings of the grating, on the krypton line at 434.31 nm.
That line is dispersed to the same place on the detector as this
wavelength is by the ultraviolet rulings, in the ratio of the two ruling
densities, so it is this wavelength that the first channel is focused at.

The line was chosen because it lands where the Rowland circle crosses the
flat detector, and because a window of N-BK7 could be found whose focus
shift there matches that of the magnesium fluoride filter in the
ultraviolet, which let the instrument be focused with the filter's effect
included.
"""

wavelength_focus_last = (600 / 2200) * 668 * u.nm
"""
The wavelength at which the last channel of the instrument was focused,
from the neon line at 668 nm.

See :data:`wavelength_focus_first`.
"""

translation_focus = 0.8733 * u.mm
"""
The displacement of the feed optic array along the axis of the instrument
which focuses the first channel, found by :func:`focus`.

It is close to the focus shift of the visible-blind filter, since the
instrument images the virtual image of the Sun onto the detector at very
nearly unit magnification, so the array must move away from the grating by
about as much as the filter moves the focus away from it.
"""

angle_focus = -0.00296 * u.deg
"""
The rotation of the feed optic array about the first feed optic which
focuses the last channel, found by :func:`focus`.

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
from ._focus import width_line, focus  # noqa: E402
from ._design import design_proposed, design  # noqa: E402
