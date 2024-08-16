import pytest
import optika._tests.test_mixins
import furst_optics._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.FrontAperture(),
    ],
)
class TestFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
    furst_optics._components_test.AbstractTestAbstactComponent,
):
    def test_surface(self, a: furst_optics.SolarDisk):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
