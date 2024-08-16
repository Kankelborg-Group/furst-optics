import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "FeedOptic",
]


@dataclasses.dataclass(eq=False, repr=False)
class FeedOptic(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    furst_optics.abcs.AbstractRowlandComponent,
    optika.mixins.Translatable,
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
        import furst_optics

        # Define the Rowland circle
        rowland_radius = 1000 * u.mm
        a = na.linspace(0, 360, axis="angle", num=1001) * u.deg
        rowland_circle = rowland_radius * na.Cartesian3dVectorArray(
            x=np.sin(a),
            z=np.cos(a),
        )

        # Define the exaggerated feed optic array
        feed_optic = furst_optics.feed_optics.FeedOptic(
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

    name: str = 'feed optic'
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

    margin_polishing: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The height above and below the clear aperture needed to 
    hold the optic for polishing.
    """

    margin_mounting: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The length of the optic used to hold it in it's mount.
    """

    material: None | optika.materials.AbstractMaterial = None
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
    def transformation(self) -> na.transformations.AbstractTransformation:
        t_center = na.transformations.Cartesian3dTranslation(
            x=0 * u.mm,
            y=0 * u.mm,
            z=-self.radius,
        )
        t_yaw = na.transformations.Cartesian3dRotationY(
            angle=-self.rowland_azimuth,
        )
        t_img = na.transformations.Cartesian3dTranslation(
            x=0 * u.mm,
            y=0 * u.mm,
            z=self.radius / 2,
        )
        return t_img @ super().transformation @ t_yaw @ t_center

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
                    y=(self.aperture_height / 2 + self.margin_mounting + self.margin_polishing),
                ),
                samples_wire=1001,
                transformation=na.transformations.Cartesian3dTranslation(
                    y=self.margin_polishing - self.margin_mounting,
                )
            ),
            transformation=self.transformation,
        )
