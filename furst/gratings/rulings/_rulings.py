import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst

# defined in the package __init__ so that Sphinx documents them
from . import depths_simulated, angle_simulated
from . import serial_numbers_delivered, angles_delivered

__all__ = [
    "efficiency_simulated",
    "rulings_simulated",
    "efficiency_delivered",
    "rulings_delivered",
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

    This is the prediction Zeiss made before the gratings were
    manufactured. No measurement of the efficiency of the delivered
    gratings exists, but Zeiss later simulated it from their measured
    groove profiles, :func:`efficiency_delivered`, which supersedes this.

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


def efficiency_delivered(
    serial_number: str = "ID01",
) -> na.FunctionArray[na.SpectralDirectionalVectorArray, na.ScalarArray]:
    """
    The first-order efficiency of a delivered grating as simulated by
    Zeiss, including the coating.

    Zeiss delivered two gratings, :data:`serial_numbers_delivered`, and
    ID01 flew.
    For its final report :cite:p:`Stock2023`, Zeiss measured the groove
    profile at the center of each grating with an atomic force microscope
    after the grating was coated, 38.5 nm peak to valley on ID01 and 43.7
    nm on ID07, and computed the efficiency of that profile with a rigorous
    solver at each of :data:`angles_delivered`.
    The efficiency includes the reflectance of the coating of aluminum and
    magnesium fluoride, and is the mean of the two polarizations, which is
    what an unpolarized source sees.
    The report shows the profile varying little across the grating, by
    about one point of efficiency between the center and the extremes.

    This supersedes :func:`efficiency_simulated`, which is the prediction
    Zeiss made before the gratings were manufactured.
    The curves are digitized from the plots in the report.

    Parameters
    ----------
    serial_number
        The serial number of the grating, one of
        :data:`serial_numbers_delivered`.

    Examples
    --------
    Plot the simulated efficiency of both gratings at each angle of
    incidence.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(
                ncols=2,
                sharey=True,
                figsize=(8, 4),
                constrained_layout=True,
            )
            for ax, serial_number in zip(axs, furst.gratings.rulings.serial_numbers_delivered):
                efficiency = furst.gratings.rulings.efficiency_delivered(serial_number)
                na.plt.plot(
                    efficiency.inputs.wavelength,
                    efficiency.outputs,
                    ax=ax,
                    axis="wavelength",
                    label=efficiency.inputs.direction,
                )
                ax.set_title(serial_number)
                ax.set_xlabel(f"wavelength ({efficiency.inputs.wavelength.unit:latex_inline})")
            axs[0].set_ylabel("efficiency")
            axs[0].legend(title="angle of incidence")
    """
    if serial_number not in serial_numbers_delivered:
        raise ValueError(
            f"serial_number must be one of {serial_numbers_delivered}, "
            f"got {serial_number!r}"
        )

    data = np.loadtxt(
        fname=pathlib.Path(__file__).parent
        / f"_data/zeiss-final-report-2023-Jul-{serial_number}.txt",
        unpack=True,
    )
    wavelength = data[0]
    efficiency = data[1:].T / 100

    return na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=na.ScalarArray(wavelength << u.nm, axes="wavelength"),
            direction=na.ScalarArray(angles_delivered, axes="angle"),
        ),
        outputs=na.ScalarArray(efficiency, axes=("wavelength", "angle")),
    )


def rulings_delivered(
    serial_number: str = "ID01",
    spacing: u.Quantity = 1 / (2200 / u.mm),
    diffraction_order: int = 1,
) -> optika.rulings.MeasuredRulings:
    """
    The rulings of a delivered grating, with the efficiency of the grooves
    alone taken from the Zeiss final report.

    As in :func:`rulings_simulated`, the coating is a separate surface
    property in the instrument model, so the efficiency here is
    :func:`efficiency_delivered` divided by the reflectance of the coating
    Zeiss assumed, :func:`furst.gratings.materials.coating_simulated`, at
    each angle of incidence.
    The final report does not give the thicknesses of the coating, so they
    are taken to be those of the design report.

    The efficiency is interpolated linearly between the angles of
    :data:`angles_delivered`, so each channel of the instrument sees the
    efficiency at its own angle of incidence.

    Parameters
    ----------
    serial_number
        The serial number of the grating, one of
        :data:`serial_numbers_delivered`.
    spacing
        The spacing between adjacent rulings.
    diffraction_order
        The diffraction order to simulate.

    Examples
    --------
    Plot the efficiency of the grooves of the flight grating alone, at each
    angle of incidence Zeiss simulated.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        rulings = furst.gratings.rulings.rulings_delivered()
        efficiency = rulings.efficiency_measured

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                efficiency.inputs.wavelength,
                efficiency.outputs,
                ax=ax,
                axis="wavelength",
                label=efficiency.inputs.direction,
            )
            ax.set_xlabel(f"wavelength ({efficiency.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("groove efficiency");
            ax.legend(title="angle of incidence");
    """
    efficiency = efficiency_delivered(serial_number)
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
        axis_angle="angle",
    )
