import pytest
import astropy.units as u
import optika._tests.test_mixins
import furst_optics


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.SolarDisk(),
        furst_optics.SolarDisk(1000 * u.arcsec),
    ]
)
class TestSolarDisk(
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_surface(self, a: furst_optics.SolarDisk):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert isinstance(result.aperture, optika.apertures.CircularAperture)
        assert result.aperture.radius > 0