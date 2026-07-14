import pytest
from optika._tests import test_mixins
from msfc_ccd._tests.test_sensors import AbstractTestAbstractSensor
import furst._components_test
import furst


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.sensors.Sensor(),
    ],
)
class TestSensor(
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
    AbstractTestAbstractSensor,
    furst._components_test.AbstractTestAbstractRowlandComponent,
):
    pass
