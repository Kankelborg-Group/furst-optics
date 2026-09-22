from typing import Generic
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst

__all__ = [
    "FeedOptic",
]


@dataclasses.dataclass(eq=False, repr=False)
class FeedOptic(
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    furst.abc.AbstractRowlandComponent,
    Generic[furst.typevars.MaterialT],
):
    """
    Model of the FURST feed optics.

    These are tall narrow cylinders which are analogs of the
    slit used in a traditional spectrograph.
    They are necesseary to achieve the demagnification necessary
    to fit the entire Sun onto one pixel on the detector.

    Examples
    --------

    Plot an exaggerated feed optic array on top of
    a Rowland circle.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import furst

        # Define the Rowland circle
        rowland_radius = 1000 * u.mm
        a = na.linspace(0, 360, axis="angle", num=1001) * u.deg
        rowland_circle = rowland_radius * na.Cartesian3dVectorArray(
            x=np.sin(a),
            z=np.cos(a),
        )

        # Define the exaggerated feed optic array
        feed_optic = furst.feed_optics.FeedOptic(
            radius=25 * u.mm,
            aperture_subtent=30 * u.deg,
            aperture_height=10 * u.mm,
            rowland_radius=rowland_radius,
            rowland_azimuth=na.linspace(
                start=5 * u.deg,
                stop=45 * u.deg,
                axis="az",
                num=7,
            ),
        )

        # Plot the feed optic array and the
        # Rowland circle.
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            feed_optic.surface.plot(
                ax=ax,
                components=("z", "x"),
                color="tab:blue",
            )
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            na.plt.plot(
                rowland_circle,
                ax=ax,
                components=("z", "x"),
                color="black",
                linestyle="dashed",
                zorder=-10,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
    """

    name: str = "feed optic"
    """
    The human-readable name of this optic.
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The radius of curvature of the optical surface.
    """

    aperture_subtent: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angular width of the clear aperture.
    """

    aperture_height: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The physical height of the clear aperture.
    """

    twist: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The rotation of the clear aperture about the vertical axis.

    Since this rotation is about the center of curvature, it slides the
    clear aperture around the curved surface rather than tilting it, and
    so sets the direction of the reflected beam; it is what aims the beam
    at the grating.
    """

    margin_polishing: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The height above and below the clear aperture needed to 
    hold the optic for polishing.
    """

    margin_mounting: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The length of the optic used to hold it in its mount.
    """

    material: furst.typevars.MaterialT = None
    """
    The coating material used to make the optic reflective
    in the target spectral range.
    """

    rowland_radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the center of the Rowland circle to
    the virtual image of the Sun within the feed optic.
    """

    rowland_azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The azimuth of the virtual image of the Sun
    on the Rowland circle, relative to the optic axis
    of the instrument.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    Physical offset from the optic's nominal position on the
    Rowland circle.
    """

    translation_focus: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The displacement of the whole array along the axis of the instrument.

    The feed optics are mounted as a unit on a stage which slides along the
    axis of the instrument, and this is the motion used to focus the first
    channel.
    A positive displacement carries the array away from the grating, which
    lengthens the distance from the virtual image to the grating and so
    *shortens* the distance from the grating to the focus.

    See Also
    --------
    :func:`furst.instruments.focus`: Finds this displacement by raytracing.
    """

    angle_focus: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The rotation of the whole array about the first feed optic.

    The array is mounted on a second stage which pivots about the first
    feed optic, and this is the motion used to focus the remaining
    channels once the first one is focused.
    The rotation is about the vertical axis through :attr:`pivot_focus`,
    so it leaves the first feed optic where it is.

    See Also
    --------
    :func:`furst.instruments.focus`: Finds this angle by raytracing.
    """

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the vector tangent to the
    Rowland circle.
    """

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the axis of symmetry
    of the feed optic.
    """

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the vector normal to
    the Rowland circle.
    """

    @property
    def axis_channel(self) -> None | str:
        """
        The name of the logical axis along which the channels of the
        instrument are distributed, taken from :attr:`rowland_azimuth`,
        or :obj:`None` if this is a single feed optic.
        """
        shape = optika.shape(self.rowland_azimuth)
        if not shape:
            return None
        if len(shape) != 1:  # pragma: nocover
            raise ValueError(
                f"the azimuths of the array must have at most one axis, "
                f"got shape {shape}"
            )
        (axis,) = shape
        return axis

    @property
    def pivot_focus(self) -> na.Cartesian3dVectorArray:
        """
        The point that :attr:`angle_focus` rotates the array about, which is
        where the first feed optic images the Sun onto the Rowland circle.

        The virtual image of the first feed optic sits a few tens of microns
        inside this point, since the clear aperture is twisted to aim the
        beam at the grating, so the rotation moves the first channel by a
        small fraction of a micron instead of leaving it exactly fixed.
        """
        azimuth = self.rowland_azimuth
        axis = self.axis_channel
        if axis is not None:
            azimuth = azimuth[{axis: 0}]
        radius = self.rowland_radius
        x = na.as_named_array(radius * np.sin(azimuth))
        z = na.as_named_array(radius * np.cos(azimuth))
        return na.Cartesian3dVectorArray(x=x, y=0 * x, z=z)

    @property
    def transformation_focus(self) -> na.transformations.AbstractTransformation:
        """
        The rigid motion of the whole array applied by the focus mechanism,
        expressed in the global coordinate system.

        See :attr:`translation_focus` and :attr:`angle_focus`.
        """
        pivot = self.pivot_focus
        return na.transformations.TransformationList(
            [
                na.transformations.Translation(-pivot),
                na.transformations.Cartesian3dRotationY(self.angle_focus),
                na.transformations.Translation(pivot),
                na.transformations.Cartesian3dTranslation(z=self.translation_focus),
            ]
        )

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        t_center = na.transformations.Cartesian3dTranslation(
            x=0 * u.mm,
            y=0 * u.mm,
            z=-self.radius,
        )
        t_yaw = na.transformations.Cartesian3dRotationY(
            angle=-self.rowland_azimuth,
        )
        t_twist = na.transformations.Cartesian3dRotationY(
            angle=self.twist,
        )
        t_img = na.transformations.Cartesian3dTranslation(
            x=0 * u.mm,
            y=0 * u.mm,
            z=self.radius / 2,
        )
        return (
            self.transformation_focus
            @ t_img
            @ super().transformation
            @ t_twist
            @ t_yaw
            @ t_center
        )

    @property
    def transformation_image(self):
        """
        Coordinate transformation from the global coordinate system
        and the virtual image of the Sun inside the feed optic.
        """

        t_img = na.transformations.Cartesian3dTranslation(
            x=0 * u.mm,
            y=0 * u.mm,
            z=self.radius / 2,
        )
        return t_img @ self.transformation

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name=self.name,
            sag=optika.sags.CylindricalSag(
                radius=self.radius,
            ),
            material=self.material,
            aperture=optika.apertures.RectangularAperture(
                half_width=na.Cartesian2dVectorArray(
                    x=self.radius * np.sin(self.aperture_subtent / 2),
                    y=self.aperture_height / 2,
                )
            ),
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=na.Cartesian2dVectorArray(
                    x=0.99 * self.radius,
                    y=(
                        self.aperture_height / 2
                        + self.margin_mounting
                        + self.margin_polishing
                    ),
                ),
                samples_wire=1001,
                transformation=na.transformations.Cartesian3dTranslation(
                    y=self.margin_polishing - self.margin_mounting,
                ),
            ),
            transformation=self.transformation,
        )
