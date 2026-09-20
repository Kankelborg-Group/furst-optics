"""
Models and measurements of the reflective coatings.
"""

import astropy.units as u

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

# These constants are defined here, rather than beside the functions that
# use them in `_materials`, so that Sphinx documents them: `autosummary`
# finds a module attribute only when it is both assigned with a docstring
# in the source of the module being documented and named in its `__all__`.
#
# They must also be defined above the import of `_materials`, which reads
# them back out of this partially initialized package.

wavelength_design = 121.6 * u.nm
"""
The wavelength that the coating on the feed optics is optimized for,
hydrogen Lyman :math:`\\alpha`.

The coating is Acton broadband VUV coating #1200, whose reflectance
:cite:t:`ActonCatalog2001` specifies at this wavelength.
"""

reflectance_design = 0.805 * u.dimensionless_unscaled
"""
The reflectance of the coating at :data:`wavelength_design`.

:cite:t:`ActonCatalog2001` specifies 78 to 83 percent for this coating,
of which this is the midpoint. That specification is quoted at
:data:`angle_witness`.
"""

thickness_aluminum = 60 * u.nm
"""
The thickness of the aluminum layer of the coating.

Aluminum is opaque in the far ultraviolet, where its skin depth is less
than 10 nm, so any thickness above about 50 nm gives the same reflectance
and this value is not critical.
"""

width_interface = 2.73 * u.nm
"""
The effective width of the interfaces between the layers of the coating.

A perfectly smooth quarter-wave stack would reflect about 95 percent at
:data:`wavelength_design`, far more than the :data:`reflectance_design`
that the vendor specifies at :data:`angle_witness`. This width is the one
which brings the model down to that specification, and it stands in for
everything the room-temperature process loses to roughness, porosity, and
oxidation of the aluminum before it is over-coated. It is not a
measurement of the roughness.

That such losses dominate is the difference between this conventional
coating and the enhanced, hot-deposited coatings of
:cite:t:`Quijada2012`, which do reach above 90 percent.
"""

wavelength_fit_min = 120 * u.nm
"""
The shortest wavelength that :func:`coating_witness_fit` fits to.

The witness sample was measured from 120 to 600 nm, but only the far
ultraviolet is of any use to FURST, and most of the measured points lie
outside it. Fitting all of them buys accuracy in the visible at the
expense of the bandpass, so the fit is restricted to it.
"""

wavelength_fit_max = 185 * u.nm
"""
The longest wavelength that :func:`coating_witness_fit` fits to.

See :data:`wavelength_fit_min`.
"""

angle_witness = 15 * u.deg
"""
The angle of incidence at which the witness samples were measured.

:cite:t:`ActonCoatingCurve` publishes the reflectance of this coating at
15 degrees, which is the geometry the vendor measures it in, and the
tabulated specification in :cite:t:`ActonCatalog2001` describes the same
measurement as near normal incidence.

The value matters very little in any case. Between this angle and the
4.6 degrees at which the feed optics are actually used, the mean
reflectance across the bandpass moves by 0.33 percentage points, and
between 15 degrees and normal it moves by less still.

Fitting this angle rather than fixing it drives it to about 70 degrees,
which is not a real geometry. A smooth two-layer model of this coating
reflects about 95 percent at :data:`wavelength_design`, far above both
the measurement and the vendor's specification, and a steep angle is the
only parameter in such a model that can suppress the far ultraviolet
while leaving the visible high. The fitted angle absorbs that error
rather than measuring anything.
"""

from ._materials import (  # noqa: E402
    coating_design,
    coating_witness_measured,
    coating_witness_fit,
)
