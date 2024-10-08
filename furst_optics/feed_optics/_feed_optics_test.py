import pytest
import astropy.units as u
import named_arrays as na
from optika._tests import test_mixins
import furst_optics._components_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst_optics.feed_optics.FeedOptic(
            radius=3 * u.mm,
            aperture_subtent=10 * u.deg,
            aperture_height=10 * u.mm,
            rowland_radius=1000 * u.mm,
            rowland_azimuth=10 * u.deg,
        )
    ],
)
class TestFeedOptics(
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
    furst_optics._components_test.AbstractTestAbstractRowlandComponent,
):

    def test_transformation_image(self, a: furst_optics.feed_optics.FeedOptic):
        result = a.transformation_image
        assert isinstance(result, na.transformations.AbstractTransformation)
