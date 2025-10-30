import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst_optics

__all__ = [
    "Spectrograph",
]


@dataclasses.dataclass(eq=False, repr=False)
class Spectrograph(
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

    source: furst_optics.sources.AbstractSource
    """
    The light source being observed by this instrument.
    """

    front_aperture: furst_optics.apertures.FrontAperture
    """
    A model of the front aperture plate of the instrument.
    """

    feed_optic: furst_optics.feed_optics.FeedOptic
    """
    A model of the feed optic array.
    """

    grating: furst_optics.gratings.Grating
    """
    A model of the diffraction grating used to disperse light.
    """

    detector: furst_optics.detectors.Detector
    """
    A model of the imaging sensor used to measure light
    at the end of the optical system.
    """

    grid_input: optika.vectors.ObjectVectorArray
    """
    A grid of samples in wavelength, pupil, and field position used
    to trace rays through the optical system.
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
        normal_rulings = grating.rulings.spacing_(position).normalized
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
        detector = self.detector.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.spacing_(position).normalized
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
        m = grating.rulings.diffraction_order
        d = grating.rulings.spacing_(position).length
        a = self.angle_grating_input()
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

    def wavelength_physical(
        self,
        wavelength_normalized: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        Convert wavelength in normalized units (between -1 and +1),
        to wavelength in physical units using :attr:`wavelength_min`
        and :attr:`wavelength_max`.

        Parameters
        ----------
        wavelength_normalized
            An array of wavelengths in normalized units to convert.
        """
        wavelength_min = self.wavelength_min
        wavelength_max = self.wavelength_max
        wavelength_range = wavelength_max - wavelength_min
        result = wavelength_range * (wavelength_normalized + 1) / 2 + wavelength_min
        return result

    @property
    def system(self) -> optika.systems.SequentialSystem:
        """
        This spectrograph expressed as an instance of
        :class:`optika.systems.SequentialSystem`.
        """
        grid = self.grid_input.explicit.copy_shallow()
        grid.wavelength = self.wavelength_physical(grid.wavelength)

        return optika.systems.SequentialSystem(
            surfaces=[
                self.front_aperture.surface,
                self.feed_optic.surface,
                self.grating.surface,
            ],
            object=self.source.surface,
            sensor=self.detector.surface,
            grid_input=grid,
            transformation=self.transformation,
        )
