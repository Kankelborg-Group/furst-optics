import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "coating_design",
    "coating_witness_measured",
    "coating_witness_fit",
]


def coating_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed coating for the FURST feed optics, Acton Optics
    broadband VUV coating #1200.

    Since we presumably don't know the formula of this proprietary
    coating, this function uses the formula in :cite:t:`Quijada2012`.

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
        angle = na.linspace(0, 75, axis="angle", num=5) * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Initialize the FURST feed optic coating model
        coating = furst_optics.feed_optics.materials.coating_design()

        # Compute the reflectivity of the feed optics
        reflectivity = coating.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity of the feed optics vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            reflectivity,
            ax=ax,
            axis="wavelength",
            label=angle,
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
        ax.legend(title="incidence angle");
    """
    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="MgF2",
                thickness=25 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.Layer(
                chemical="Al",
                thickness=60 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=3 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
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

    This function assumes that the angle of incidence is 75 degrees from the normal.
    This is just a guess and will have to be updated with better information.

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
                direction=75 * u.deg,
            ),
            outputs=reflectivity.to(u.dimensionless_unscaled),
        ),
        substrate=optika.materials.Layer(
            chemical="SiO2",
        ),
    )

    return result


def coating_witness_fit() -> optika.materials.MultilayerMirror:
    """
    A coating fitted to the :func:`coating_witness_measured` measurement.

    Examples
    --------
    Plot the fitted vs. measured reflectivity of the feed optic witness sample.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika
        from furst_optics import feed_optics

        # Load the measured reflectivity of the witness sample
        multilayer_measured = feed_optics.materials.coating_witness_measured()
        measurement = multilayer_measured.efficiency_measured

        # Isolate the angle of incidence of the measurement
        angle_incidence = measurement.inputs.direction

        # Fit a MgF2+Al coating to the measured reflectivity
        coating = feed_optics.materials.coating_witness_fit()

        # Define the rays incident on the coating that will be used to
        # compute the reflectivity
        rays = optika.rays.RayVectorArray(
            wavelength=measurement.inputs.wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle_incidence),
                y=0,
                z=np.cos(angle_incidence),
            ),
        )

        # Compute the reflectivity of the fitted multilayer stack
        reflectivity_fit = coating.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the fitted vs. measured reflectivity
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.scatter(
            measurement.inputs.wavelength,
            measurement.outputs,
            ax=ax,
            label="measured"
        );
        na.plt.plot(
            rays.wavelength,
            reflectivity_fit,
            ax=ax,
            label="fitted",
            color="tab:orange",
        );
        ax.set_xlabel(f"wavelength ({rays.wavelength.unit:latex_inline})")
        ax.set_ylabel("reflectivity")
        ax.legend();

        # Print the fitted coating
        coating
    """

    design = coating_design()

    measurement = coating_witness_measured()
    unit = u.nm

    reflectivity = measurement.efficiency_measured.outputs
    angle_incidence = measurement.efficiency_measured.inputs.direction

    rays = optika.rays.RayVectorArray(
        wavelength=measurement.efficiency_measured.inputs.wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle_incidence),
            y=0,
            z=np.cos(angle_incidence),
        ),
    )

    normal = na.Cartesian3dVectorArray(0, 0, -1)

    def _coating(
        thickness_MgF2: float,
        thickness_Al: float,
        width_interface: float,
    ):
        result = coating_design()
        result.layers[0].thickness = thickness_MgF2 * unit
        result.layers[1].thickness = thickness_Al * unit
        result.layers[0].interface.width = width_interface * unit
        result.layers[1].interface.width = width_interface * unit
        result.substrate.interface.width = width_interface * unit

        return result

    def _func(x: np.ndarray):

        multilayer = _coating(*x)

        reflectivity_fit = multilayer.efficiency(
            rays=rays,
            normal=normal,
        )

        result = np.sqrt(np.mean(np.square(reflectivity_fit - reflectivity)))

        return result.ndarray.value

    fit = scipy.optimize.minimize(
        fun=_func,
        x0=[
            design.layers[0].thickness.to_value(unit),
            design.layers[1].thickness.to_value(unit),
            design.substrate.interface.width.to_value(unit),
        ],
        bounds=[
            (0, None),
            (0, None),
            (0, None),
        ],
    )

    return _coating(*fit.x)
