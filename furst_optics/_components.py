import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractComponent",
    "AbstractRowlandComponent",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractComponent(
    optika.mixins.Printable,
    optika.mixins.Transformable,
):

    @property
    @abc.abstractmethod
    def surface(self):
        """
        Convert this object into an instance of
        :class:`optika.surfaces.AbstractSurface`.
        """



@dataclasses.dataclass(eq=False, repr=False)
class AbstractRowlandComponent(
    optika.mixins.Transformable,
):
    """
    A base class representing an optical component
    on the Rowland circle.
    """

    @property
    @abc.abstractmethod
    def rowland_radius(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the Rowland circle.
        """

    @property
    @abc.abstractmethod
    def rowland_azimuth(self) -> u.Quantity | na.AbstractScalar:
        """
        The azimuth of the optical component on
        the Rowland circle.
        """

    @property
    def transformation(self) -> None | na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.TransformationList(
            [
                na.transformations.Cartesian3dTranslation(
                    x=0 * u.mm,
                    y=0 * u.mm,
                    z=self.rowland_radius,
                ),
                na.transformations.Cartesian3dRotationY(self.rowland_azimuth),
            ]
        )
