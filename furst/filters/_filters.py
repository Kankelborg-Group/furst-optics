from typing import Generic
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import furst

__all__ = [
    "Filter",
]


@dataclasses.dataclass(eq=False, repr=False)
class Filter(
    optika.mixins.Translatable,
    furst.abc.AbstractRowlandComponent,
    Generic[furst.typevars.MaterialT],
):
    r"""
    A model of the visible-blind filter in front of the sensor.

    The filter is a plane-parallel window mounted on the camera head,
    square to the sensor and a distance :attr:`distance` in front of it,
    so it shares the sensor's place on the Rowland circle and its
    :attr:`translation`.
    It is modeled as a pair of surfaces, a front face made of
    :attr:`material` and a back face made of :class:`optika.materials.Vacuum`
    a distance :attr:`thickness` behind it.

    A window of index :math:`n` and thickness :math:`d` in a converging
    beam moves the focus a distance :math:`d (1 - 1 / n)` away from the
    window, which :meth:`focus_shift` computes, so the sensor must be moved
    back by that amount to stay in focus.

    Examples
    --------

    Plot the focus shift of a 2 mm magnesium fluoride window across the
    FURST bandpass.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import furst

        # Define the filter model
        filter = furst.filters.Filter(
            material=furst.filters.materials.transmission_witness_measured(),
            thickness=2 * u.mm,
            radius_clear=25.4 * u.mm,
        )

        # Compute the focus shift across the bandpass
        wavelength = na.linspace(120, 185, axis="wavelength", num=101) * u.nm
        shift = filter.focus_shift(wavelength)

        # Plot the focus shift as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(wavelength, shift, ax=ax);
            ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
            ax.set_ylabel(f"focus shift ({shift.unit:latex_inline})");
    """

    name: str = "filter"
    """
    The human-readable name of this optic.
    """

    material: furst.typevars.MaterialT = None
    """
    The material of the front face of the filter, which carries the
    transmission of the coating and the index of refraction of the window.
    """

    thickness: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The thickness of the window.
    """

    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the clear aperture of the window.
    """

    radius_mech: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the window itself.
    """

    distance: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the front face of the window to the sensor,
    measured along the normal of the sensor.
    """

    rowland_radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the center of the Rowland circle to
    the center of the sensor.
    """

    rowland_azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The azimuth of the center of the sensor
    on the Rowland circle, relative to the optic axis
    of the instrument.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    An additional translation vector, shared with the sensor.
    """

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dTranslation(
            z=-self.distance,
        )

    def _surface(
        self,
        name: str,
        material: optika.materials.AbstractMaterial,
        z: u.Quantity | na.AbstractScalar,
    ) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name=name,
            material=material,
            aperture=optika.apertures.CircularAperture(
                radius=self.radius_clear,
            ),
            aperture_mechanical=optika.apertures.CircularAperture(
                radius=self.radius_mech,
            ),
            transformation=self.transformation
            @ na.transformations.Cartesian3dTranslation(z=z),
        )

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        The front face of the window, which carries :attr:`material`.
        """
        return self._surface(
            name=self.name,
            material=self.material,
            z=0 * u.mm,
        )

    @property
    def surface_back(self) -> optika.surfaces.Surface:
        """
        The back face of the window, through which the light leaves the
        window and returns to vacuum.
        """
        return self._surface(
            name=f"{self.name}_back",
            material=optika.materials.Vacuum(),
            z=self.thickness,
        )

    @property
    def surfaces(self) -> list[optika.surfaces.Surface]:
        """
        Both faces of the window, in the order the light meets them.
        """
        return [self.surface, self.surface_back]

    def focus_shift(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        r"""
        The distance that this window moves a focus behind it,
        :math:`d (1 - 1 / n)` for a window of thickness :math:`d` and
        index of refraction :math:`n`.

        This is exact for the ray along the normal of the window, and the
        variation across an :math:`f/7.5` beam is a few microns.

        Parameters
        ----------
        wavelength
            The wavelength at which to evaluate the index of refraction.
        """
        rays = optika.rays.RayVectorArray(wavelength=wavelength)
        n = self.material.index_refraction(rays)
        return (self.thickness * (1 - 1 / n)).to(u.mm)
