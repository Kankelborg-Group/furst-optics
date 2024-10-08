import astropy.units as u
import optika
from optika._tests import test_mixins
import furst_optics


class AbstractTestAbstactComponent(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
):
    def test_surface(self, a: furst_optics.abc.AbstractComponent):
        result = a.surface
        assert isinstance(result, optika.surfaces.AbstractSurface)


class AbstractTestAbstractRowlandComponent(
    AbstractTestAbstactComponent,
):
    def test_rowland_radius(self, a: furst_optics.abc.AbstractRowlandComponent):
        assert a.rowland_radius >= 0 * u.mm

    def test_rowland_azimuth(self, a: furst_optics.abc.AbstractRowlandComponent):
        assert a.rowland_azimuth.unit.is_equivalent(u.deg)
