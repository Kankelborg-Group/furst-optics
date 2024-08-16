import pytest
import optika._tests.test_mixins
import furst_optics._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.apertures.FrontAperture(),
    ],
)
class TestFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
    furst_optics._components_test.AbstractTestAbstactComponent,
):
    pass
