import dataclasses
import numpy as np
import astropy.units as u
import sunpy.sun.constants
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "SolarDisk",
]


@dataclasses.dataclass(eq=False, repr=False)
class SolarDisk(
    optika.mixins.Translatable,
    furst_optics.abcs.AbstractComponent,
):
    """
    The nominal scene observed by FURST, the entire solar disk.
    """

    radius: None | u.Quantity | na.AbstractScalar = None
    """
    The radius of the solar disk observed by FURST.
    If :obj:`None` (the default), 
    :obj:`sunpy.sun.constants.average_angular_size` is used.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """Offset of the solar disk on the celestial sphere."""

    def __post_init__(self):
        if self.radius is None:
            self.radius = sunpy.sun.constants.average_angular_size

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="solar disk",
            aperture=optika.apertures.CircularAperture(
                radius=np.cos(self.radius),
            ),
            is_field_stop=True,
            transformation=self.transformation,
        )
