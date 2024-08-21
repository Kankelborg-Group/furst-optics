import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "coating_design",
    "coating_witness_measured",
]


def coating_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed coating for the FURST feed optics, Acton Optics
    broadband VUV coating #1200.

    Examples
    --------

    Plot the efficiency of the coating across the VUV range

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import furst_optics

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(120, 600, axis="wavelength", num=1001) * u.nm

        # Define the incident rays from the wavelength array
        angle = 15 * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Initialize the FURST feed optic material
        material = furst_optics.feed_optics.materials.coating_design()

        # Compute the reflectivity of the feed optics
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity of the feed optics vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
    """
    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="MgF2",
                thickness=25 * u.nm,
                # interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.Layer(
                chemical="Al",
                thickness=50 * u.nm,
                # interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=3 * u.mm,
            # interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )


def coating_witness_measured() -> optika.materials.MeasuredMirror:
    """
    A reflectivity measurement of the witness samples to the
    feed optics.

    Examples
    --------
    Load the witness sample measurement and plot it as a function
    of wavelength against the modeled reflectivity.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import optika
        import furst_optics

        # Load the coating model
        coating_model = furst_optics.feed_optics.materials.coating_design()

        # Load the model and the witness sample measurements
        coating_measurement = furst_optics.feed_optics.materials.coating_witness_measured()
        measurement = coating_measurement.efficiency_measured

        # Isolate the wavelengths of the measurement
        wavelength = measurement.inputs.wavelength

        # Isolate the incidence angle of the measurement
        angle = measurement.inputs.direction

        # Calculate the reflectivity of the model for the same
        # wavelengths as the measurements
        reflectivity_model = coating_model.efficiency(
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

        # Plot the measurement as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                wavelength,
                reflectivity_model,
                axis="wavelength",
                ax=ax,
                label="model",
            )
            na.plt.plot(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
                label="measurement",
            )
            ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
            ax.set_ylabel("reflectivity");
            ax.legend();
    """
    wavelength, reflectivity = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/witness-2023-May-24.txt",
        skiprows=1,
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength")
    reflectivity = na.ScalarArray(reflectivity << u.percent, axes="wavelength")

    result = optika.materials.MeasuredMirror(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=15 * u.deg,
            ),
            outputs=reflectivity,
        ),
        substrate=optika.materials.Layer(
            chemical="SiO2",
        ),
    )

    return result
