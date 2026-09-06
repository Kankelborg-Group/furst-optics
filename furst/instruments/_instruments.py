import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst

__all__ = [
    "Instrument",
]


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

        return optika.systems.SequentialSystem(
            surfaces=[
                self.front_aperture.surface,
                self.feed_optic.surface,
                self.grating.surface,
            ],
            object=self.source.surface,
            sensor=self.camera.surface,
            grid_input=grid,
            transformation=self.transformation,
        )
