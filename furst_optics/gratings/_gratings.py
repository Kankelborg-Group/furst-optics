from typing import Generic, TypeVar
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "SagT",
    "MaterialT",
    "RulingT",
    "Grating",
]

#: Generic sag type
SagT = TypeVar("SagT", bound=None | optika.sags.AbstractSag)

#: Generic material type
MaterialT = TypeVar("MaterialT", bound=None | optika.materials.AbstractMaterial)

#: Generic ruling type
RulingT = TypeVar("RulingT", bound=None | optika.rulings.AbstractRulings)


@dataclasses.dataclass(eq=False, repr=False)
class Grating(
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    furst_optics.abc.AbstractRowlandComponent,
    Generic[SagT, MaterialT, RulingT],
):
    """
    A model of the FURST diffraction grating.

    This is a concave, spherical diffraction grating
    a rectangular aperture.

    Examples
    --------

    Plot an exaggerated grating on the Rowland circle.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import furst_optics

        # Define the Rowland circle
        rowland_radius = 1000 * u.mm
        a = na.linspace(0, 360, axis="angle", num=1001) * u.deg
        rowland_circle = rowland_radius * na.Cartesian3dVectorArray(
            x=np.sin(a),
            z=np.cos(a),
        )

        # Define the grating model
        grating = furst_optics.gratings.Grating(
            sag=optika.sags.SphericalSag(
                radius=-2 * rowland_radius,
            ),
            width_clear=na.Cartesian2dVectorArray(
                x=1000 * u.mm,
                y=20 * u.mm,
            ),
            material=optika.materials.Mirror(),
            rowland_radius=rowland_radius,
            rowland_azimuth=175 * u.deg,
        )

        # Plot the grating surface and the Rowland circle
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            grating.surface.plot(
                ax=ax,
                components=("z", "x"),
                color="tab:blue",
            )
            na.plt.plot(
                rowland_circle,
                ax=ax,
                components=("z", "x"),
                color="black",
                linestyle="dashed",
                zorder=-10,
            )
            ax.set_aspect("equal")

    """

    name: str = "grating"
    """
    The human-readable name of this optic.
    """

    sag: SagT = None
    """
    The sag profile of the grating surface.
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of curvature of the optical surface.
    """

    width_clear: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    """
    The height and width of the clear aperture of the grating.
    """

    width_mech: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    """
    The height and width of the grating substrate.
    """

    material: MaterialT = None
    """
    The coating material used to make the optic reflective
    in the target spectral range.
    """

    rulings: RulingT = None
    """
    A model of the grating ruling spacing and profile.
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
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name=self.name,
            sag=self.sag,
            material=self.material,
            aperture=optika.apertures.RectangularAperture(
                half_width=self.width_clear / 2,
            ),
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.width_mech / 2,
            ),
            rulings=self.rulings,
            is_pupil_stop=True,
            transformation=self.transformation,
        )
