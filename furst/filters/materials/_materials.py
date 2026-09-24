import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

# defined in the package __init__ so that Sphinx documents it
from . import angle_witness

__all__ = [
    "transmission_design",
    "transmission_witness_measured",
]


def transmission_design() -> optika.materials.MeasuredFilter:
    """
    The transmission of the visible-blind filter as the vendor publishes it.

    The filter is an Acton 120-VBB solar-blind coating
    :cite:p:`ActonSolarBlind` on a window of magnesium fluoride.
    Acton publishes the transmission of its standard filter, on a 5 mm
    window, at normal incidence from 120 to 420 nm, and this is that curve,
    digitized from the published plot.
    The FURST windows are thinner, :data:`furst.filters.thickness_design`,
    but magnesium fluoride absorbs little in the bandpass, so the published
    curve is used as it stands.

    As for :func:`transmission_witness_measured`, the curve is the
    transmission of the whole filter, so the returned material applies it
    once at the front face of the window, and the window refracts but does
    not absorb again.
    Below the shortest wavelength of the curve, 121.4 nm, the transmission
    is held at its value there.

    Examples
    --------
    Plot the published transmission against the witness sample measured for
    the flight filter.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        published = furst.filters.materials.transmission_design()
        measured = furst.filters.materials.transmission_witness_measured()

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            for material, label in [
                (published, "published"),
                (measured, "witness sample"),
            ]:
                measurement = material.efficiency_measured
                na.plt.plot(
                    measurement.inputs.wavelength,
                    measurement.outputs,
                    ax=ax,
                    label=label,
                )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})");
            ax.set_ylabel("transmission");
            ax.legend();
    """
    wavelength, transmission = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/acton-120-vbb-published.txt",
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength")
    transmission = na.ScalarArray(transmission, axes="wavelength")

    return optika.materials.MeasuredFilter(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=0 * u.deg,
            ),
            outputs=transmission,
        ),
        medium=optika.materials.Dielectric("MgF2"),
        is_medium_measured=True,
    )


def transmission_witness_measured() -> optika.materials.MeasuredFilter:
    """
    A transmission measurement of the witness sample to the visible-blind
    filter.

    The filter is an Acton 120-VBB solar-blind coating
    :cite:p:`ActonSolarBlind` on a 2 mm window of magnesium fluoride,
    which passes the far ultraviolet and rejects the visible light that the
    sensor would otherwise respond to.
    The coating is proprietary, so this measurement, and the curve the
    vendor publishes, :func:`transmission_design`, are the only models of it.

    The measurement is the transmission of the whole coated witness piece,
    so it includes the absorption of the magnesium fluoride and the
    reflection from its uncoated back face, and the returned material
    applies it once at the front face of the window.
    The window is modeled as :class:`optika.materials.Dielectric` magnesium
    fluoride, whose tabulated index of refraction bends the rays and shifts
    the focus, and whose tabulated absorption is not applied again.
    The thickness of the witness piece was not recorded, so if it differed
    from the flight window the absorption near 120 nm differs by an
    unknown amount.

    Note that this sample transmits more than the vendor's published curve,
    :func:`transmission_design`, which peaks at 21 percent near 187 nm for
    the standard 5 mm window, where this sample peaks at 24 percent near
    171 nm.

    Examples
    --------
    Load the witness sample measurement and plot the transmission as a
    function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        # Load the witness sample measurement
        filter = furst.filters.materials.transmission_witness_measured()
        measurement = filter.efficiency_measured

        # Plot the measurement as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
            )
            ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("transmission");
    """
    wavelength, transmission = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/witness-2023-May-08.txt",
        skiprows=1,
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength")
    transmission = na.ScalarArray(transmission << u.percent, axes="wavelength")

    return optika.materials.MeasuredFilter(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=angle_witness,
            ),
            outputs=transmission.to(u.dimensionless_unscaled),
        ),
        medium=optika.materials.Dielectric("MgF2"),
        is_medium_measured=True,
    )
