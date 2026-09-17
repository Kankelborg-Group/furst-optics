import numpy as np
import astropy.units as u
import named_arrays as na
import furst


def test_design():
    result = furst.instruments.design(
        num_field=3,
        num_pupil=3,
    )
    assert isinstance(result, furst.instruments.Instrument)

    # the parameters of the original design study
    assert result.feed_optic.rowland_azimuth.shape == {"channel": 7}
    assert result.grating.rowland_radius == 675 * u.mm
    assert result.grating.sag.radius == -1350 * u.mm
    assert result.grating.width_clear.x == 180 * u.mm
    assert result.feed_optic.radius == 3 * u.mm
    assert result.camera.sensor.rowland_radius == result.grating.rowland_radius
    assert result.feed_optic.rowland_radius == result.grating.rowland_radius

    # every channel lands on the sensor
    rays = result.system.rayfunction_default
    assert np.isfinite(rays.outputs.position.x).all()
    assert (na.nominal(rays.outputs.intensity).mean("wavelength") > 0).all()
