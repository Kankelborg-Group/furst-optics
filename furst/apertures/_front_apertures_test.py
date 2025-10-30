import pytest
import optika._tests.test_mixins
import furst._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.apertures.FrontAperture(),
    ],
)
class TestFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
    furst._components_test.AbstractTestAbstactComponent,
):
    pass
