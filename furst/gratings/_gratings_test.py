import pytest
import astropy.units as u
import optika
from optika._tests import test_mixins
import furst._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.gratings.Grating(
            sag=optika.sags.SphericalSag(
                radius=1000 * u.mm,
            ),
            width_clear=10 * u.mm,
            width_mech=15 * u.mm,
            material=optika.materials.Mirror(),
            rulings=optika.rulings.Rulings(
                spacing=1 * u.um,
                diffraction_order=1,
            ),
        )
    ],
)
class TestGrating(
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
    furst._components_test.AbstractTestAbstractRowlandComponent,
):
    pass
