import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst

# defined in the package __init__ so that Sphinx documents them
from . import depths_simulated, angle_simulated

__all__ = [
    "efficiency_simulated",
    "rulings_simulated",
]


def efficiency_simulated(
    depth: u.Quantity = 42 * u.nm,
) -> na.FunctionArray[na.SpectralDirectionalVectorArray, na.ScalarArray]:
    """
    The first-order efficiency of the grating as simulated by Zeiss,
    including the coating.

    In its design report :cite:p:`Burkhardt2021`, Zeiss computed the
    efficiency of the etched, quasi-sinusoidal profile it expected to
    manufacture, at each of :data:`depths_simulated`, with a rigorous
    (C-method) solver. The profile is coated with 70 nm of aluminum and
    27 nm of magnesium fluoride, and the efficiency includes the
    reflectance of that coating, so it is the fraction of the incident
    light that leaves in the first order. The simulation is at
    :data:`angle_simulated` and is the mean of the two polarizations,
    which is what an unpolarized source sees.

    No measurement of the efficiency of the delivered grating exists,
    so this simulation is the best available estimate of it.

    Parameters
    ----------
    depth
        The depth of the groove profile, one of :data:`depths_simulated`.

    Examples
    --------
    Plot the simulated efficiency at each of the three profile depths.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            for depth in furst.gratings.rulings.depths_simulated:
                efficiency = furst.gratings.rulings.efficiency_simulated(depth)
                na.plt.plot(
                    efficiency.inputs.wavelength,
                    efficiency.outputs,
                    ax=ax,
                    label=depth,
                )
            ax.set_xlabel(f"wavelength ({efficiency.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("efficiency");
            ax.legend(title="profile depth");
    """
    depths = depths_simulated.to_value(u.nm)
    index = np.flatnonzero(np.isclose(depths, depth.to_value(u.nm)))
    if index.size != 1:
        raise ValueError(
            f"depth must be one of {depths_simulated}, got {depth}",
        )
    index = int(index[0])

    data = np.loadtxt(
        fname=pathlib.Path(__file__).parent
        / "_data/zeiss-simulated-efficiency-2021-May.txt",
        unpack=True,
    )
    wavelength = data[0]
    te = data[1 + 2 * index]
    tm = data[2 + 2 * index]
    efficiency = (te + tm) / 2 / 100

    return na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=na.ScalarArray(wavelength << u.nm, axes="wavelength"),
            direction=angle_simulated,
        ),
        outputs=na.ScalarArray(efficiency, axes="wavelength"),
    )


def rulings_simulated(
    depth: u.Quantity = 42 * u.nm,
    spacing: u.Quantity = 1 / (2200 / u.mm),
    diffraction_order: int = 1,
) -> optika.rulings.MeasuredRulings:
    """
    The rulings of the grating, with the efficiency of the grooves alone
    taken from the Zeiss simulation.

    :func:`efficiency_simulated` is the efficiency of the grooves and the
    coating together, but in the instrument model the coating is a
    separate surface property, :func:`furst.gratings.materials.coating_measured`,
    and the two are multiplied. So the efficiency here is the simulated
    efficiency divided by the reflectance of the coating that Zeiss
    assumed in the simulation,
    :func:`furst.gratings.materials.coating_simulated`, which leaves the
    fraction of the reflected light that the grooves send into the first
    order. That fraction is between 30 and 33 percent across the bandpass
    and varies smoothly, as it should.

    Combining it with the measured coating rather than the assumed one
    lowers the short end of the bandpass, where the coating as deposited
    reflects less than the smooth stack Zeiss assumed.

    Parameters
    ----------
    depth
        The depth of the groove profile, one of :data:`depths_simulated`.
    spacing
        The spacing between adjacent rulings.
    diffraction_order
        The diffraction order to simulate.

    Examples
    --------
    Plot the efficiency of the grooves alone against the simulated
    efficiency of grooves and coating together.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        efficiency = furst.gratings.rulings.efficiency_simulated()
        rulings = furst.gratings.rulings.rulings_simulated()

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                efficiency.inputs.wavelength,
                efficiency.outputs,
                ax=ax,
                label="grooves and coating, Zeiss",
            )
            na.plt.plot(
                rulings.efficiency_measured.inputs.wavelength,
                rulings.efficiency_measured.outputs,
                ax=ax,
                label="grooves alone",
            )
            ax.set_xlabel(f"wavelength ({efficiency.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("efficiency");
            ax.legend();
    """
    efficiency = efficiency_simulated(depth)
    wavelength = efficiency.inputs.wavelength
    angle = efficiency.inputs.direction

    coating = furst.gratings.materials.coating_simulated()
    reflectance = coating.efficiency(
        rays=optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        ),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )

    return optika.rulings.MeasuredRulings(
        spacing=spacing,
        diffraction_order=diffraction_order,
        efficiency_measured=na.FunctionArray(
            inputs=efficiency.inputs,
            outputs=efficiency.outputs / reflectance,
        ),
    )
