import pytest
import optika
from msfc_ccd._tests.test_cameras import AbstractTestAbstractSensor
import furst


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.cameras.Camera(),
    ],
)
class TestCameras(
    AbstractTestAbstractSensor,
):
    def test_surface(
        self,
        a: furst.cameras.Camera,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)
