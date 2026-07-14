import pytest
import optika
from msfc_ccd._tests.test_cameras import AbstractTestAbstractCamera
import furst


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.cameras.Camera(),
    ],
)
class TestCameras(
    AbstractTestAbstractCamera,
):
    def test_surface(
        self,
        a: furst.cameras.Camera,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)
