import pytest
import astropy.units as u
import optika._tests.test_mixins
import furst._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.sources.SolarDisk(),
        furst.sources.SolarDisk(1000 * u.arcsec),
    ],
)
class TestSolarDisk(
    optika._tests.test_mixins.AbstractTestTranslatable,
    furst._components_test.AbstractTestAbstactComponent,
):
    def test_surface(self, a: furst.sources.SolarDisk):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert isinstance(result.aperture, optika.apertures.CircularAperture)
        assert result.aperture.radius > 0
