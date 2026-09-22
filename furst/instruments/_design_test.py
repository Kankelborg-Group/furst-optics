import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst


@pytest.mark.parametrize(
    argnames="func,rowland_radius",
    argvalues=[
        (furst.instruments.design_proposed, 675 * u.mm),
        (furst.instruments.design, 677 * u.mm),
    ],
)
def test_design(func, rowland_radius: u.Quantity):
    result = func(
        num_field=3,
        num_pupil=3,
    )
    assert isinstance(result, furst.instruments.Instrument)

    # the parameters of the original design study
    assert result.feed_optic.rowland_azimuth.shape == {"channel": 7}
    assert result.grating.rowland_radius == rowland_radius
    assert result.grating.sag.radius == -2 * rowland_radius
    assert result.grating.width_clear.x == 180 * u.mm
    assert result.feed_optic.radius == 3 * u.mm
    assert result.camera.sensor.rowland_radius == result.grating.rowland_radius
    assert result.feed_optic.rowland_radius == result.grating.rowland_radius

    # the measured coatings are on the feed optic and the grating, and the
    # grating has the simulated groove efficiency
    assert isinstance(result.feed_optic.material, optika.materials.MeasuredMirror)
    assert isinstance(result.grating.material, optika.materials.MeasuredMirror)
    assert isinstance(result.grating.rulings, optika.rulings.MeasuredRulings)

    # the visible-blind filter is a coated magnesium fluoride window on the
    # camera head, and nothing compensates for its focus shift yet: the
    # sensor stays on the Rowland circle
    assert isinstance(result.filter.material, optika.materials.MeasuredFilter)
    assert isinstance(result.filter.material.medium, optika.materials.Dielectric)
    assert result.filter.rowland_radius == result.camera.sensor.rowland_radius
    assert result.filter.rowland_azimuth == result.camera.sensor.rowland_azimuth
    assert np.all(result.camera.sensor.translation == 0 * u.mm)
    assert np.all(result.filter.translation == result.camera.sensor.translation)

    # the filter is centered on the beam from the grating and normal to it,
    # a couple of inches in front of the sensor
    origin = na.Cartesian3dVectorArray() * u.mm
    position_grating = result.grating.transformation(origin)
    position_sensor = result.camera.sensor.transformation(origin)
    position_filter = result.filter.transformation(origin)
    beam = position_sensor - position_grating
    beam = beam / beam.length
    offset = position_sensor - position_filter
    normal = result.filter.transformation.transformation_linear(
        na.Cartesian3dVectorArray(0, 0, 1)
    )
    assert np.isclose(offset.length, result.filter.distance)
    assert np.isclose(offset @ beam, result.filter.distance)
    assert np.isclose(normal @ beam, 1)
    assert 3 * u.deg < np.abs(result.filter.yaw) < 7 * u.deg

    # every channel lands on the sensor, and the coatings, rulings, and
    # filter have cost it most of the light
    rays = result.system.rayfunction_default
    assert np.isfinite(rays.outputs.position.x).all()
    intensity = na.nominal(rays.outputs.intensity)
    assert (intensity.mean("wavelength") > 0).all()
    assert (intensity < 0.1).all()
