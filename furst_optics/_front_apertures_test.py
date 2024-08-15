import pytest
import optika._tests.test_mixins
import furst_optics


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.FrontAperture(),
    ],
)
class TestFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_surface(self, a: furst_optics.SolarDisk):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
