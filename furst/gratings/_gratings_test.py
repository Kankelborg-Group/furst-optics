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


@pytest.mark.parametrize(
    argnames="serial_number",
    argvalues=furst.gratings.rulings.serial_numbers_delivered,
)
def test_width_delivered(serial_number: str):
    """
    Each delivered grating has its ruled area and substrate recorded, and
    the ruled area, flanked by the alignment gratings, fits well inside
    the substrate.
    """
    width_clear = furst.gratings.width_clear_delivered[serial_number]
    width_mech = furst.gratings.width_mech_delivered[serial_number]
    assert 180 * u.mm < width_clear.x < width_mech.x - 5 * u.mm
    assert 30 * u.mm < width_clear.y < width_mech.y - 20 * u.mm
    assert abs(width_mech.x - 190 * u.mm) < 0.1 * u.mm
    assert abs(width_mech.y - 60 * u.mm) < 0.1 * u.mm


def test_width_delivered_keys():
    """Both dictionaries cover exactly the delivered gratings."""
    serial_numbers = set(furst.gratings.rulings.serial_numbers_delivered)
    assert set(furst.gratings.width_clear_delivered) == serial_numbers
    assert set(furst.gratings.width_mech_delivered) == serial_numbers
