import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "wavelength_design",
    "reflectance_design",
    "angle_witness",
    "coating_design",
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

thickness_aluminum = 60 * u.nm
"""
The thickness of the aluminum layer of the coating.

Aluminum is opaque in the far ultraviolet, where its skin depth is less
than 10 nm, so any thickness above about 50 nm gives the same reflectance
and this value is not critical.
"""

reflectance_design = 0.805 * u.dimensionless_unscaled
"""
The reflectance of the coating at :data:`wavelength_design`.

:cite:t:`ActonCatalog2001` specifies 78 to 83 percent for this coating
at normal incidence, of which this is the midpoint.
"""

width_interface = 2.7 * u.nm
"""
The effective width of the interfaces between the layers of the coating.

A perfectly smooth quarter-wave stack would reflect about 95 percent at
:data:`wavelength_design`, far more than the
:data:`reflectance_design` that the vendor specifies. This width is the
one which brings the model down to that specification, and it stands in
for everything the room-temperature process loses to roughness,
porosity, and oxidation of the aluminum before it is over-coated. It is
not a measurement of the roughness.

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

angle_witness = 0 * u.deg
"""
The angle of incidence at which the witness samples were measured.

The geometry was not recorded, but normal incidence is both the natural
reading and very nearly free of consequence.

:cite:t:`ActonCatalog2001` quotes this coating at normal incidence, and
the sample reads 81.5 percent at 120 nm, inside the 78 to 83 percent
specified there. More importantly, the reflectance barely depends on the
angle over the range that matters: between normal incidence and the 4.6
degrees at which the feed optics are actually used, the mean reflectance
across the bandpass moves by 0.04 percentage points, against a
measurement noise of 0.18.

Fitting this angle instead of fixing it drives it to about 70 degrees,
which is not a real geometry. A smooth two-layer model of this coating
reflects about 95 percent at :data:`wavelength_design`, some 13 points
above both the measurement and the vendor's own specification, and a
steep angle is the only parameter in such a model that can suppress the
far ultraviolet while leaving the visible high. The fitted angle absorbs
that error rather than measuring anything, so the geometry cannot be
inferred from a model which is known to be wrong by far more than the
effect being inferred.
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
    :data:`reflectance_design` that :cite:t:`ActonCatalog2001`
    specifies.

    The vendor publishes two numbers about this coating, the wavelength
    it is optimized for and its reflectance there, and this model has
    two free parameters, so it is determined rather than fitted. The
    thickness it implies agrees with the one
    :func:`coating_witness_fit` recovers independently from the witness
    sample.

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
    magnesium_fluoride = optika.chemicals.Chemical("MgF2")

    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical=magnesium_fluoride,
                thickness=_thickness_quarter_wave(
                    chemical=magnesium_fluoride,
                    wavelength=wavelength_design,
                ),
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width_interface,
                ),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.Layer(
                chemical="Al",
                thickness=thickness_aluminum,
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width_interface,
                ),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=3 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(
                width_interface,
            ),
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

    The measurement is taken to be at :data:`angle_witness`, since
    :cite:t:`ActonCatalog2001` quotes the reflectance of this coating at
    normal incidence. The measured reflectance at 120 and 125 nm, 81.5
    and 83.2 percent, falls inside the 78 to 83 percent that the vendor
    specifies at 121.6 nm, which supports that reading.

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
                direction=angle_witness,
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

    For radiometry, prefer :func:`coating_witness_measured` itself, which
    interpolates the measured reflectance directly. This fit is a smooth
    two-layer stand-in for a proprietary coating and cannot reproduce the
    measurement to better than about 1.5 percentage points across the
    bandpass, against a measurement noise of 0.18. It is useful for
    extrapolating outside the measured wavelengths, and for comparing
    against :func:`coating_design`, but it does not improve on the
    measurement where the measurement exists.

    The thickness of the magnesium fluoride and the width of the
    interfaces are fitted, over the wavelengths between
    :data:`wavelength_fit_min` and :data:`wavelength_fit_max`.

    The thickness of the aluminum is not fitted, since aluminum is
    opaque in this wavelength range and the measurement cannot constrain
    it. Fitting it anyway admits a second, unphysical solution with
    semi-transparent aluminum, which fits the far ultraviolet slightly
    better at the cost of the visible.

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

    design = coating_design()

    measurement = coating_witness_measured()
    unit = u.nm

    wavelength = measurement.efficiency_measured.inputs.wavelength
    angle_incidence = measurement.efficiency_measured.inputs.direction

    # only the wavelengths the instrument actually uses
    where = (wavelength > wavelength_fit_min) & (wavelength < wavelength_fit_max)
    wavelength = wavelength[where]
    reflectivity = measurement.efficiency_measured.outputs[where]

    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle_incidence),
            y=0,
            z=np.cos(angle_incidence),
        ),
    )

    normal = na.Cartesian3dVectorArray(0, 0, -1)

    def _coating(
        thickness_MgF2: float,
        width_interface: float,
    ):
        result = coating_design()
        result.layers[0].thickness = thickness_MgF2 * unit
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
            design.substrate.interface.width.to_value(unit),
        ],
        bounds=[
            (0, None),
            (0, None),
        ],
    )

    return _coating(*fit.x)
