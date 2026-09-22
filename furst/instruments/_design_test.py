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
    # camera head, and the sensor and filter have been moved back together
    # to compensate for the focus shift of the window
    assert isinstance(result.filter.material, optika.materials.MeasuredFilter)
    assert isinstance(result.filter.material.medium, optika.materials.Dielectric)
    assert result.filter.rowland_radius == result.camera.sensor.rowland_radius
    assert result.filter.rowland_azimuth == result.camera.sensor.rowland_azimuth
    assert 0.5 * u.mm < result.camera.sensor.translation.z < 0.9 * u.mm
    assert result.filter.translation == result.camera.sensor.translation

    # every channel lands on the sensor, and the coatings, rulings, and
    # filter have cost it most of the light
    rays = result.system.rayfunction_default
    assert np.isfinite(rays.outputs.position.x).all()
    intensity = na.nominal(rays.outputs.intensity)
    assert (intensity.mean("wavelength") > 0).all()
    assert (intensity < 0.1).all()
