import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "coating_measured",
    "coating_simulated",
]


def coating_measured() -> optika.materials.MeasuredMirror:
    """
    A reflectivity measurement of the coating on the FURST grating.

    The grating was coated by Zeiss, its manufacturer, with aluminum
    protected by magnesium fluoride. Before coating the grating, Zeiss
    deposited the same coating on a test piece and measured its
    reflectivity from 120 to 230 nm, and this is that measurement. The
    only record of it is a plot, so the smoothed curve of that plot was
    digitized to a 1 nm grid; the digitization is good to about a tenth
    of a nanometer in wavelength and a few tenths of a percentage point
    in reflectivity.

    The angle of incidence of the measurement was not recorded, so it is
    taken to be normal. The grating is used at 10 to 18 degrees, and the
    reflectance of an aluminum and magnesium fluoride coating changes by
    only a small fraction of a percentage point over that range.

    This is the reflectivity of the coating alone, not the efficiency of
    the grating, which is the product of this reflectivity and the
    fraction of the light the rulings send into the first order.

    Examples
    --------
    Load the measurement and plot it as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import furst

        # Load the measurement of the grating coating
        coating = furst.gratings.materials.coating_measured()
        measurement = coating.efficiency_measured

        # Plot the measurement as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
            )
            ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("reflectivity");
    """
    wavelength, reflectivity = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/zeiss-test-coating-2023-May.txt",
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength")
    reflectivity = na.ScalarArray(reflectivity, axes="wavelength")

    return optika.materials.MeasuredMirror(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=0 * u.deg,
            ),
            outputs=reflectivity,
        ),
        substrate=optika.materials.Layer(
            chemical="SiO2",
        ),
    )


def coating_simulated() -> optika.materials.MultilayerMirror:
    """
    The coating on the grating as Zeiss assumed it when simulating the
    efficiency of the grating.

    The design report :cite:p:`Burkhardt2021` coats the simulated groove
    profile with 70 nm of aluminum and 27 nm of magnesium fluoride over
    fused silica, with perfectly smooth interfaces. This is that stack.
    Its only use is to divide the coating back out of the simulated
    efficiency in :func:`furst.gratings.rulings.rulings_simulated`, so
    that the measured coating, :func:`coating_measured`, can take its
    place.

    Examples
    --------
    Plot the reflectance of the assumed coating against the measured one.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import furst

        # Load the measured and the assumed coatings
        measured = furst.gratings.materials.coating_measured()
        simulated = furst.gratings.materials.coating_simulated()

        # Evaluate the assumed coating at the wavelengths of the measurement,
        # at the angle of incidence Zeiss simulated
        measurement = measured.efficiency_measured
        angle = furst.gratings.rulings.angle_simulated
        reflectance = simulated.efficiency(
            rays=optika.rays.RayVectorArray(
                wavelength=measurement.inputs.wavelength,
                direction=na.Cartesian3dVectorArray(
                    x=np.sin(angle),
                    y=0,
                    z=np.cos(angle),
                ),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
                label="measured",
            )
            na.plt.plot(
                measurement.inputs.wavelength,
                reflectance,
                ax=ax,
                label="assumed by Zeiss",
            )
            ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("reflectance");
            ax.legend();
    """
    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="MgF2",
                thickness=27 * u.nm,
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.Layer(
                chemical="Al",
                thickness=70 * u.nm,
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=35 * u.mm,
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )
