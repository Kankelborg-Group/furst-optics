import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import msfc_ccd
import furst

__all__ = [
    "Sensor",
]


@dataclasses.dataclass(repr=False)
class Sensor(
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    msfc_ccd.TeledyneCCD230,
    furst.abc.AbstractRowlandComponent,
):
    """A model of the CCD sensors used to detect light."""

    rowland_radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the center of the Rowland circle to
    the center of the detector.
    """

    rowland_azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The azimuth of the center of the detector
    on the Rowland circle, relative to the optic axis
    of the instrument.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """An additional translation vector."""

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The pitch angle of this sensor."""

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The yaw angle of this sensor."""

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The roll angle of this sensor."""

    @property
    def surface(self) -> optika.sensors.AbstractImagingSensor:
        """Represent this object as an :mod:`optika` surface."""
        return optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=self.width_pixel,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=self.num_pixel_active,
            material=optika.sensors.materials.e2v_ccd97(
                temperature=self.temperature,
            ),
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.width_package / 2,
            ),
            transformation=self.transformation,
        )
