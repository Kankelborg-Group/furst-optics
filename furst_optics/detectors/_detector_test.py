import pytest
import astropy.units as u
import named_arrays as na
from optika._tests import test_mixins
import furst_optics._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.detectors.Detector(
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray(
                x="detector_x",
                y="detector_y",
            ),
            num_pixel=2048,
        )
    ],
)
class TestDetector(
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
    furst_optics._components_test.AbstractTestAbstractRowlandComponent,
):
    pass
