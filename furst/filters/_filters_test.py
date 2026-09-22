import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.filters.Filter(
            material=furst.filters.materials.transmission_witness_measured(),
            thickness=2 * u.mm,
            radius_clear=25 * u.mm,
            radius_mech=25.4 * u.mm,
            distance=50 * u.mm,
            rowland_radius=677 * u.mm,
            rowland_azimuth=5 * u.deg,
        ),
    ],
)
class TestFilter:
    def test_surfaces(self, a: furst.filters.Filter):
        surfaces = a.surfaces
        assert len(surfaces) == 2
        assert surfaces[0] is not surfaces[1]
        for surface in surfaces:
            assert isinstance(surface, optika.surfaces.Surface)
        assert surfaces[0].material is a.material
        assert isinstance(surfaces[1].material, optika.materials.Vacuum)

        # the back face is a window's thickness behind the front face,
        # along the normal of the front face
        origin = na.Cartesian3dVectorArray() * u.mm
        front = surfaces[0].transformation(origin)
        back = surfaces[1].transformation(origin)
        assert np.isclose((back - front).length, a.thickness)

    def test_transformation(self, a: furst.filters.Filter):
        """
        The filter sits in front of a sensor at the same place on the
        Rowland circle, by the given distance.
        """
        sensor = furst.sensors.Sensor(
            rowland_radius=a.rowland_radius,
            rowland_azimuth=a.rowland_azimuth,
            translation=a.translation,
        )
        origin = na.Cartesian3dVectorArray() * u.mm
        position_filter = a.transformation(origin)
        position_sensor = sensor.transformation(origin)
        assert np.isclose((position_sensor - position_filter).length, a.distance)

        # the filter is closer to the center of the Rowland circle than the
        # sensor, on the side facing the grating
        assert position_filter.length < position_sensor.length

    def test_focus_shift(self, a: furst.filters.Filter):
        wavelength = na.linspace(120, 185, axis="wavelength", num=11) * u.nm
        shift = a.focus_shift(wavelength)
        assert (shift > 0.5 * u.mm).all()
        assert (shift < 0.9 * u.mm).all()

        # the index falls with wavelength, so the shift does too
        assert (np.diff(shift, axis="wavelength") < 0).all()
