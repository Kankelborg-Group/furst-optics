import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "FrontAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class FrontAperture(
    optika.mixins.Translatable,
    furst_optics.abcs.AbstractComponent,
):
    """
    The front aperture plate of the FURST instrument.

    This plate is both the entrance aperture to the optical system
    and the mechanical interface between the optical table and the
    rocket skins.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    The physical location of the front aperture plate relative to the rest
    of the optical system.
    """

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="front aperture",
            transformation=self.transformation,
        )
