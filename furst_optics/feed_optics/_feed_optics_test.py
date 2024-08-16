import pytest
import astropy.units as u
import furst_optics._rowland_components_test


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
    furst_optics._rowland_components_test.AbstractTestAbstractRowlandComponent
):
    pass
