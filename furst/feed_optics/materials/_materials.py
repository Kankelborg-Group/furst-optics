import functools
import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "wavelength_design",
    "reflectance_design",
    "angle_specification",
    "thickness_aluminum",
    "width_interface",
    "coating_design",
    "angle_witness",
    "coating_witness_measured",
    "coating_witness_fit",
]

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
of which this is the midpoint.
"""

angle_specification = 0 * u.deg
"""
The angle of incidence at which :data:`reflectance_design` is specified.

:cite:t:`ActonCatalog2001` quotes this coating at normal incidence.
Note that this is the geometry of the vendor's *specification*, and says
nothing about the geometry of the witness measurement, which is
:func:`angle_witness`.
"""

thickness_aluminum = 60 * u.nm
"""
The thickness of the aluminum layer of the coating.

Aluminum is opaque in the far ultraviolet, where its skin depth is less
than 10 nm, so any thickness above about 50 nm gives the same reflectance
and this value is not critical.
"""

width_interface = 2.7 * u.nm
"""
The effective width of the interfaces between the layers of the coating.

A perfectly smooth quarter-wave stack would reflect about 95 percent at
:data:`wavelength_design`, far more than the :data:`reflectance_design`
that the vendor specifies. This width is the one which brings the model
down to that specification, and it stands in for everything the
room-temperature process loses to roughness, porosity, and oxidation of
the aluminum before it is over-coated. It is not a measurement of the
roughness.

That such losses dominate is the difference between this conventional
coating and the enhanced, hot-deposited coatings of
:cite:t:`Quijada2012`, which do reach above 90 percent.
"""


def _thickness_quarter_wave(
    chemical: optika.chemicals.AbstractChemical,
    wavelength: u.Quantity,
) -> u.Quantity:
    """
    The thickness of a quarter-wave layer of the given chemical.

    Parameters
    ----------
    chemical
        The material of the layer.
    wavelength
        The wavelength at which the layer is a quarter of a wave thick.
    """
    # the index comes back as a named array, and a layer thickness is a
    # plain quantity
    index = na.as_named_array(np.real(chemical.n(wavelength))).ndarray
    return (wavelength / (4 * index)).to(u.nm)


def _coating(
    thickness_magnesium_fluoride: u.Quantity,
    width: u.Quantity,
) -> optika.materials.MultilayerMirror:
    """
    A magnesium-fluoride-over-aluminum coating with the given thickness
    and interface width.

    Parameters
    ----------
    thickness_magnesium_fluoride
        The thickness of the protective magnesium fluoride layer.
    width
        The width of every interface in the stack.
    """

    def interface():
        return optika.materials.profiles.ErfInterfaceProfile(width)

    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="MgF2",
                thickness=thickness_magnesium_fluoride,
                interface=interface(),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.Layer(
                chemical="Al",
                thickness=thickness_aluminum,
                interface=interface(),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=3 * u.mm,
            interface=interface(),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )


def _efficiency(
    coating: optika.materials.AbstractMultilayerMirror,
    wavelength: u.Quantity | na.AbstractScalar,
    angle: u.Quantity,
) -> na.AbstractScalar:
    """
    The reflectance of a coating at the given wavelength and angle of
    incidence.

    Parameters
    ----------
    coating
        The coating to evaluate.
    wavelength
        The wavelengths at which to evaluate it.
    angle
        The angle of incidence, measured from the surface normal.
    """
    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )
    return coating.efficiency(
        rays=rays,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )


def coating_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed coating for the FURST feed optics, Acton broadband
    VUV coating #1200.

    This is a conventional aluminum mirror protected by magnesium
    fluoride. Since the recipe is proprietary, this function models it as
    the simplest coating consistent with what the vendor publishes: a
    quarter wave of magnesium fluoride at :data:`wavelength_design`, over
    aluminum thick enough to be opaque, with the interfaces broadened by
    :data:`width_interface` so that the stack reflects the
    :data:`reflectance_design` that :cite:t:`ActonCatalog2001` specifies
    at :data:`angle_specification`.

    The vendor publishes two numbers about this coating, the wavelength
    it is optimized for and its reflectance there, and this model has two
    free parameters, so it is determined rather than fitted.

    Note that this is *not* the enhanced coating of
    :cite:t:`Quijada2012`, whose magnesium fluoride is deposited on a
    heated substrate and which reaches above 90 percent at 121.6 nm. The
    feed optics carry the conventional room-temperature coating.

    Examples
    --------

    Plot the efficiency of the coating across the VUV range

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import furst

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
        coating = furst.feed_optics.materials.coating_design()

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
    return _coating(
        thickness_magnesium_fluoride=_thickness_quarter_wave(
            chemical=optika.chemicals.Chemical("MgF2"),
            wavelength=wavelength_design,
        ),
        width=width_interface,
    )


def _witness_data() -> na.FunctionArray:
    """
    The measured reflectance of the witness samples against wavelength,
    without any assumption about the geometry it was measured in.
    """
    wavelength, reflectivity = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/witness-2023-May-24.txt",
        skiprows=1,
        unpack=True,
    )
    return na.FunctionArray(
        inputs=na.ScalarArray(wavelength << u.nm, axes="wavelength"),
        outputs=na.ScalarArray(reflectivity << u.percent, axes="wavelength").to(
            u.dimensionless_unscaled
        ),
    )


@functools.cache
def _fit_witness() -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """
    Solve for the coating and the geometry that best reproduce the
    witness measurement.

    The thickness of the magnesium fluoride, the width of the interfaces,
    and the angle of incidence of the measurement are all free, and are
    fitted to every measured wavelength.

    The angle is fitted because it was never recorded. It is strongly
    constrained by the shape of the measured curve, since the
    interference structure moves with angle, and much more weakly by the
    overall level: :cite:t:`ActonCatalog2001` notes that reflectance at
    45 degrees is normally within 2 to 4 percent of normal incidence, so
    the level alone cannot determine it.

    The thickness of the aluminum is not fitted, since aluminum is opaque
    in this wavelength range and the measurement cannot constrain it.
    Fitting it anyway admits a second, unphysical solution with
    semi-transparent aluminum.

    Returns
    -------
    The magnesium fluoride thickness, the interface width, and the angle
    of incidence.
    """
    data = _witness_data()
    design = coating_design()

    unit_length = u.nm
    unit_angle = u.deg

    def _unpack(x: np.ndarray):
        thickness = x[0] * unit_length
        width = x[1] * unit_length
        angle = x[2] * unit_angle
        return thickness, width, angle

    def _objective(x: np.ndarray) -> float:
        thickness, width, angle = _unpack(x)
        efficiency = _efficiency(
            coating=_coating(thickness, width),
            wavelength=data.inputs,
            angle=angle,
        )
        residual = np.sqrt(np.mean(np.square(efficiency - data.outputs)))
        return float(na.as_named_array(residual).ndarray)

    # The objective has a shallow second minimum at small angles, about
    # twice the residual of the true one, which a local search started
    # near normal incidence will fall into. Start from a spread of angles
    # and keep the best.
    fit = min(
        (
            scipy.optimize.minimize(
                fun=_objective,
                x0=[
                    design.layers[0].thickness.to_value(unit_length),
                    width_interface.to_value(unit_length),
                    angle_start,
                ],
                bounds=[
                    (0, None),
                    (0, None),
                    (0, 89),
                ],
            )
            for angle_start in [0, 30, 60, 80]
        ),
        key=lambda result: result.fun,
    )

    return _unpack(fit.x)


def angle_witness() -> u.Quantity:
    """
    The angle of incidence at which the witness samples were measured.

    The angle was never recorded, so it is solved for along with the
    coating by :func:`coating_witness_fit`. See :func:`_fit_witness` for
    why it can be recovered from the measurement at all.

    Examples
    --------

    .. jupyter-execute::

        import furst

        furst.feed_optics.materials.angle_witness()
    """
    thickness, width, angle = _fit_witness()
    return angle


def coating_witness_measured() -> optika.materials.MeasuredMirror:
    """
    A reflectivity measurement of the witness samples to the
    feed optics.

    The geometry of the measurement was not recorded, so the angle of
    incidence reported here is :func:`angle_witness`, which is solved for
    rather than assumed.

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
        import furst

        # Load the coating model
        coating_model = furst.feed_optics.materials.coating_design()

        # Load the model and the witness sample measurements
        coating_measurement = furst.feed_optics.materials.coating_witness_measured()
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
    data = _witness_data()

    return optika.materials.MeasuredMirror(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=data.inputs,
                direction=angle_witness(),
            ),
            outputs=data.outputs,
        ),
        substrate=optika.materials.Layer(
            chemical="SiO2",
        ),
    )


def coating_witness_fit() -> optika.materials.MultilayerMirror:
    """
    A coating fitted to the :func:`coating_witness_measured` measurement.

    The thickness of the magnesium fluoride, the width of the interfaces,
    and the angle of incidence of the measurement are fitted jointly. The
    fitted angle is available separately as :func:`angle_witness`.

    Examples
    --------
    Plot the fitted vs. measured reflectivity of the feed optic witness sample.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika
        from furst import feed_optics

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
    thickness, width, angle = _fit_witness()
    return _coating(thickness, width)
