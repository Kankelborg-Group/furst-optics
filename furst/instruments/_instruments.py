import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst

__all__ = [
    "Instrument",
]


def _width_focus(
    instrument: "Instrument",
    wavelength: na.AbstractScalar,
    channel: int,
    num_pupil: int,
) -> na.AbstractScalar:
    """
    The width of the lines of one channel, combined in quadrature.

    This is the quantity each stage of the focus minimizes, so that a
    channel is focused as a whole rather than at one wavelength.

    Parameters
    ----------
    instrument
        The instrument to trace rays through.
    wavelength
        The wavelengths to measure, whose axes are the ones combined over.
    channel
        The index of the channel to measure.
    num_pupil
        The number of samples along each axis of the pupil.
    """
    width = instrument.width_line(wavelength, num_pupil)
    width = width[{instrument.feed_optic.axis_channel: channel}]
    axis = tuple(na.shape(wavelength))
    if axis:
        width = np.sqrt(np.square(width).mean(axis))
    return width


def _vertex(
    inputs: na.AbstractScalar,
    outputs: na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    """
    The position of the minimum of a parabola fitted to a focus curve.

    The square of the width of the line is the quantity which is quadratic
    in the defocus, so it is what is fitted, as on the bench.

    Parameters
    ----------
    inputs
        The sampled positions of the feed optic array.
    outputs
        The width of the line measured at each position.
    axis
        The logical axis along which the curve is sampled.
    """
    fit = na.PolynomialFitFunctionArray.from_degree(
        inputs=inputs,
        outputs=np.square(outputs),
        degree=2,
        axis_polynomial=axis,
        center=inputs.mean(axis),
    )
    name_linear, name_quadratic = fit.coefficient_names[1:]
    coefficients = fit.coefficients.components
    return fit.center - coefficients[name_linear] / (2 * coefficients[name_quadratic])


@dataclasses.dataclass(eq=False, repr=False)
class Instrument(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
):
    """
    A generic model of the FURST instrument,
    which includes the feed optics, grating, visible-blind filter,
    and detector.
    """

    name: str
    """
    The human-readable name of this optics model.
    """

    source: furst.sources.AbstractSource
    """
    The light source being observed by this instrument.
    """

    front_aperture: furst.apertures.FrontAperture
    """
    A model of the front aperture plate of the instrument.
    """

    feed_optic: furst.feed_optics.FeedOptic
    """
    A model of the feed optic array.
    """

    grating: furst.gratings.Grating
    """
    A model of the diffraction grating used to disperse light.
    """

    camera: furst.cameras.Camera
    """A model of the camera and sensors used to measure light."""

    wavelength: u.Quantity | na.AbstractScalar
    """
    A default grid of wavelengths to trace through the system.

    Can be either in normalized coordinates (in the range :math:`-1` to :math:`+1`)
    or in physical coordinates (with units of length).

    See Also
    --------
    :attr:`wavelength_physical`: This value converted into in physical coordinates.
    """

    field: na.AbstractCartesian2dVectorArray
    """
    A default grid of field positions to trace through the system.
    
    Can be either in normalized coordinates (in the range :math:`-1` to :math:`+1`)
    or in physical coordinates (with units of angle).
    """

    pupil: na.AbstractCartesian2dVectorArray
    """
    A default grid of pupil positions to trace through the system.
    
    Can be either in normalized coordinates (in the range :math:`-1` to :math:`+1`)
    or in physical coordinates (with units of length).
    """

    filter: None | furst.filters.Filter = None
    """
    A model of the visible-blind filter in front of the sensor.

    If :obj:`None`, no filter is modeled.
    """

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    Rotation about the vector perpendicular to the optic axis and the
    vector normal to the optical table.
    """

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    Rotation of the instrument about the vector normal to 
    the FURST optical table.
    """

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    Rotation of the instrument about the optic axis.
    """

    @property
    def angle_grating_input(self) -> na.AbstractScalar:
        """
        The angle between the grating normal and the vector
        pointing from the center of the grating
        to the virtual image of the source inside the feed optic.
        """
        feed_optic = self.feed_optic
        grating = self.grating
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.spacing_(position, normal_surface).normalized
        transformation_grating = grating.transformation.inverse
        transformation_feed_optic = feed_optic.transformation_image
        transformation = transformation_grating @ transformation_feed_optic
        x = na.Cartesian3dVectorArray() * u.mm
        x = transformation(x)
        return np.arctan2(
            x @ normal_rulings,
            x @ normal_surface,
        )

    def angle_grating_output(self, axis: str):
        """
        The angles between the grating normal and the vectors
        pointing from the center of the grating
        to the edges of the detector.

        Parameters
        ----------
        axis
            An axis name for the array of points
            around the edges of the detector.
        """
        detector = self.camera.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.spacing_(position, normal_surface).normalized
        transformation = grating.transformation.inverse @ detector.transformation
        wire = np.moveaxis(
            a=detector.aperture.wire(),
            source="wire",
            destination=axis,
        )
        wire = transformation(wire)
        return np.arctan2(
            wire @ normal_rulings,
            wire @ normal_surface,
        )

    def _wavelength_test_grid(
        self,
        axis_output: str,
    ) -> na.AbstractScalar:
        """
        A grid of test wavelengths used to find the wavelength
        range for each channel of FURST.

        Parameters
        ----------
        axis_output
            An axis name for the array of points
            around the edges of the detector.
        """
        position = na.Cartesian3dVectorArray() * u.mm
        grating = self.grating.surface
        normal = grating.sag.normal(position)
        m = grating.rulings.diffraction_order
        d = grating.rulings.spacing_(position, normal).length
        a = self.angle_grating_input
        b = self.angle_grating_output(axis_output)
        result = np.abs((np.sin(a) + np.sin(b)) * d / m)
        return result.to(u.AA)

    @property
    def wavelength_min(self) -> u.Quantity | na.AbstractScalar:
        """The minimum wavelength for each channel."""
        axis = "_dummy"
        wavelength = self._wavelength_test_grid(axis)
        return wavelength.min(axis)

    @property
    def wavelength_max(self) -> u.Quantity | na.AbstractScalar:
        """The maximum wavelength for each channel."""
        axis = "_dummy"
        wavelength = self._wavelength_test_grid(axis)
        return wavelength.max(axis)

    @property
    def wavelength_physical(self) -> na.AbstractScalar:
        """
        A normalized version of :attr:`wavelength` guaranteed to be in
        physical units.
        """
        wavelength = self.wavelength
        if na.unit_normalized(wavelength).is_equivalent(u.dimensionless_unscaled):
            wavelength_min = self.wavelength_min
            wavelength_max = self.wavelength_max
            wavelength_range = wavelength_max - wavelength_min
            wavelength = wavelength_range * (wavelength + 1) / 2 + wavelength_min
        return wavelength

    @property
    def system(self) -> optika.systems.SequentialSystem:
        """
        This spectrograph expressed as an instance of
        :class:`optika.systems.SequentialSystem`.
        """
        grid = optika.vectors.ObjectVectorArray(
            wavelength=self.wavelength_physical,
            field=self.field,
            pupil=self.pupil,
        )

        surfaces = [
            self.front_aperture.surface,
            self.feed_optic.surface,
            self.grating.surface,
        ]
        if self.filter is not None:
            surfaces += self.filter.surfaces

        return optika.systems.SequentialSystem(
            surfaces=surfaces,
            object=self.source.surface,
            sensor=self.camera.surface,
            grid_input=grid,
            transformation=self.transformation,
        )

    def width_line(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        num_pupil: int = 11,
    ) -> na.AbstractScalar:
        """
        The width of the spectral line formed at the given wavelength,
        measured along the dispersion direction, for every channel.

        This is the quantity minimized to focus the instrument, the analogue
        of the width of a calibration lamp line measured on the bench.
        It is computed from a point source at the center of the field, so it
        is the blur of the optics alone and not the disk-integrated line
        spread function.
        Channels which do not see the given wavelength catch no rays, and
        their width is :obj:`numpy.nan`.

        Parameters
        ----------
        wavelength
            The wavelength of the line, in physical units.
            Any axes of this array are carried through to the result, so a
            grid of wavelengths can be measured at once.
        num_pupil
            The number of samples along each axis of the pupil.

        Examples
        --------

        Plot the focus curve of the first channel, the width of a line as
        the feed optic array slides along the axis of the instrument.

        .. jupyter-execute::

            import dataclasses
            import matplotlib.pyplot as plt
            import astropy.units as u
            import astropy.visualization
            import named_arrays as na
            import furst

            # Load the design, and a wavelength in its first channel
            instrument = furst.instruments.design()
            wavelength = 121.6 * u.nm

            # Slide the feed optic array along the axis of the instrument
            translation = na.linspace(-1, 2, axis="position", num=13) * u.mm
            feed_optic = dataclasses.replace(
                instrument.feed_optic,
                translation_focus=translation,
            )
            instrument = dataclasses.replace(instrument, feed_optic=feed_optic)

            # Measure the width of the line in the first channel
            axis_channel = instrument.feed_optic.axis_channel
            width = instrument.width_line(wavelength)
            width = width[{axis_channel: 0}].to(u.um)

            # Plot the focus curve
            with astropy.visualization.quantity_support():
                fig, ax = plt.subplots(constrained_layout=True)
                na.plt.plot(translation, width, ax=ax, marker="o");
                ax.set_xlabel(f"displacement of the array ({translation.unit:latex_inline})");
                ax.set_ylabel(f"width of the line ({width.unit:latex_inline})");
        """
        instrument = dataclasses.replace(
            self,
            wavelength=wavelength,
            field=na.Cartesian2dVectorArray(0, 0),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=num_pupil,
                centers=True,
            ),
        )

        rays = instrument.system.rayfunction_default.outputs

        axis = ("pupil_x", "pupil_y")
        weight = rays.unvignetted.astype(float)
        position = rays.position.x

        # a channel which sees none of the given wavelength catches no rays,
        # and divides zero by zero to give the documented NaN
        with np.errstate(invalid="ignore"):
            mean = (position * weight).sum(axis) / weight.sum(axis)
            variance = (np.square(position - mean) * weight).sum(axis)
            variance = variance / weight.sum(axis)
            result = np.sqrt(variance)

        return result

    def focused(
        self,
        wavelength: None | na.AbstractScalar = None,
        translation: None | na.AbstractScalar = None,
        angle: None | na.AbstractScalar = None,
        num_pupil: int = 11,
    ) -> "Instrument":
        """
        A copy of this instrument with its feed optic array moved to focus
        it, reproducing the procedure carried out during assembly.

        The array is mounted on two stages, and each is used to focus one
        end of the spectrum:

        #. The lower stage slides the whole array along the axis of the
           instrument until the first channel is sharpest.
        #. The upper stage pivots the array about the first feed optic,
           which leaves the first channel where it is, until the last
           channel is sharpest.

        The remaining channels are not adjusted; they land wherever the
        mechanism puts them, as they do on the bench.

        A channel is focused as a whole rather than at one wavelength: the
        widths of the lines at ``wavelength`` are combined in quadrature,
        which balances the channel across the detector, since a flat
        detector meets the curved focal surface at only two points.
        The bench instead focused a single calibration line in each of the
        two channels, chosen to fall where the detector crosses the Rowland
        circle, which is the same idea carried out with the lines that were
        available.

        Each step samples the width over a grid of stage positions and fits
        a parabola to its square, and the grid is traced all at once, so
        the whole procedure costs two raytraces.

        Parameters
        ----------
        wavelength
            The wavelengths to focus each channel over.
            Normalized coordinates give every channel its own range, which
            is the default: five wavelengths evenly spaced across the range
            this instrument traces.
        translation
            The displacements of the array to sample in the first step.
        angle
            The rotations of the array to sample in the second step.
        num_pupil
            The number of samples along each axis of the pupil.

        Examples
        --------

        Focus the design and print the positions found.

        .. jupyter-execute::

            import furst

            instrument = furst.instruments.design().focused()

            feed_optic = instrument.feed_optic
            print(f"the array slides {feed_optic.translation_focus.ndarray:+.4f}")
            print(f"the array pivots {feed_optic.angle_focus.ndarray:+.5f}")
        """
        axis = "position"

        if wavelength is None:
            wavelength = na.linspace(
                start=self.wavelength.min(),
                stop=self.wavelength.max(),
                axis="wavelength",
                num=5,
            )
        if translation is None:
            translation = na.linspace(-1, 2, axis=axis, num=13) * u.mm
        if angle is None:
            angle = na.linspace(-0.3, 0.3, axis=axis, num=13) * u.deg

        feed_optic = self.feed_optic

        def moved(translation_focus, angle_focus):
            return dataclasses.replace(
                self,
                feed_optic=dataclasses.replace(
                    feed_optic,
                    translation_focus=translation_focus,
                    angle_focus=angle_focus,
                ),
            )

        # the lower stage, which focuses the first channel
        translation_focus = _vertex(
            inputs=translation,
            outputs=_width_focus(
                instrument=moved(translation, 0 * u.deg),
                wavelength=wavelength,
                channel=0,
                num_pupil=num_pupil,
            ),
            axis=axis,
        )

        # the upper stage, which focuses the last channel
        angle_focus = _vertex(
            inputs=angle,
            outputs=_width_focus(
                instrument=moved(translation_focus, angle),
                wavelength=wavelength,
                channel=~0,
                num_pupil=num_pupil,
            ),
            axis=axis,
        )

        return moved(translation_focus, angle_focus)
