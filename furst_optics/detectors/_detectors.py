import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "Detector",
]


@dataclasses.dataclass(eq=False, repr=False)
class Detector(
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    furst_optics.abc.AbstractRowlandComponent,
):
    """
    A model of FURST's imaging sensor and camera.

    These are manufactured by Marshall Space Flight Center
    for use with FURST.

    Examples
    --------

    Plot the surface of the detector and the Rowland circle.

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
        detector = furst_optics.detectors.Detector(
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray(
                x="detector_x",
                y="detector_y",
            ),
            num_pixel=4096,
            material=optika.materials.Mirror(),
            rowland_radius=rowland_radius,
            rowland_azimuth=10 * u.deg,
        )

        # Plot the grating surface and the Rowland circle
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                rowland_circle,
                ax=ax,
                components=("z", "x"),
                color="black",
                linestyle="dashed",
                zorder=-10,
            )
            detector.surface.plot(
                ax=ax,
                components=("z", "x"),
                color="tab:orange",
            )
            ax.set_aspect("equal")
            ax.set_xlim(960 * u.mm, 1020 * u.mm)
            ax.set_ylim(120 * u.mm, 220 * u.mm)

    """

    name: str = "detector"
    """
    The human-readable name of this detector.
    """

    manufacturer: str = ""
    """
    The company that manufactured this detector.
    """

    model_number: str = ""
    """
    The model number of this detector.
    """

    serial_number: str = ""
    """
    The unique serial number associated with this detector.
    """

    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    """
    The physical width of a pixel for this detector.
    """

    axis_pixel: None | na.Cartesian2dVectorArray[str, str] = None
    """
    The name of each axis of the pixel array.
    """

    num_pixel: int | na.Cartesian2dVectorArray[int, int] = 0
    """
    The number of pixels along each axis of the pixel array.
    """

    num_pixel_overscan: int = 0
    """
    The number of overscan columns for each tap.
    """

    num_pixel_blank: int = 0
    """
    The number of blank columns for each tap.
    """

    material: None | optika.sensors.AbstractImagingSensorMaterial = None
    """
    A model of the light-sensitive material of this detector.
    """

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
    """
    Physical offset from the optic's nominal position on the
    Rowland circle.
    """

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the long axis of the detector.
    """

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the short axis of the detector.
    """

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the vector normal to
    the surface of the detector.
    """

    temperature: u.Quantity | na.ScalarArray = 0 * u.K
    """
    The operating temperature of this detector.
    """

    gain: u.Quantity | na.ScalarArray = 0 * u.electron / u.DN
    """
    The ratio between the number of electrons measured by the sensor
    and the data number reported by the ADC
    """

    readout_noise: u.Quantity | na.ScalarArray = 0 * u.DN
    """
    The standard deviation of the noise introduced by the readout
    electronics.
    """

    dark_current: u.Quantity | na.ScalarArray = 0 * u.electron / u.s
    """
    The amount of dark signal measured by the detector at the current
    temperature. 
    """

    charge_diffusion: u.Quantity | na.AbstractScalar = 0 * u.um
    """
    The standard deviation of the charge diffusion kernel.
    """

    timedelta_transfer: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The time required to transfer an image from the light-sensitive
    area of the detector to the masked area.
    """

    timedelta_readout: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The time required to digitize an image collected by the sensor.
    """

    timedelta_exposure: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The current exposure time of this detector.
    """

    timedelta_exposure_min: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The minimum exposure time allowed by this detector.
    """

    timedelta_exposure_max: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The maximum exposure time allowed by this detector.
    """

    bits_adc: int = 0
    """
    The number of bits collected by the analog-to-digital converter
    on this detector.
    """

    @property
    def surface(self) -> optika.sensors.ImagingSensor:
        return optika.sensors.ImagingSensor(
            name=self.name,
            width_pixel=self.width_pixel,
            axis_pixel=self.axis_pixel,
            num_pixel=self.num_pixel,
            timedelta_exposure=self.timedelta_exposure,
            material=self.material,
            transformation=self.transformation,
        )
