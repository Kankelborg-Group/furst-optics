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

    # the nominal coatings are on the feed optic and the grating, and the
    # grating has the simulated groove efficiency
    assert isinstance(result.feed_optic.material, optika.materials.MultilayerMirror)
    assert isinstance(result.grating.material, optika.materials.MultilayerMirror)
    assert isinstance(result.grating.rulings, optika.rulings.MeasuredRulings)

    # the visible-blind filter is a coated magnesium fluoride window on the
    # camera head, and nothing compensates for its focus shift yet: the
    # sensor stays on the Rowland circle
    assert isinstance(result.filter.material, optika.materials.MeasuredFilter)
    assert isinstance(result.filter.material.medium, optika.materials.Dielectric)
    assert result.filter.thickness == furst.filters.thickness_design
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


def test_as_built():
    design = furst.instruments.design(num_field=3, num_pupil=3)
    result = furst.instruments.as_built(num_field=3, num_pupil=3)
    assert isinstance(result, furst.instruments.Instrument)

    # the flight grating has a longer radius than the layout it sits in,
    # which is otherwise the layout of the design
    assert result.grating.serial_number == "ID01"
    assert result.grating.sag.radius == -1359 * u.mm
    assert result.grating.rowland_radius == design.grating.rowland_radius
    assert result.grating.rowland_azimuth == design.grating.rowland_azimuth
    assert result.camera.sensor.rowland_radius == design.camera.sensor.rowland_radius
    assert np.all(
        result.feed_optic.rowland_azimuth == design.feed_optic.rowland_azimuth
    )

    # its clear aperture is the area Zeiss ruled on it, a little wider and
    # much taller than the design needs
    width_clear = result.grating.width_clear
    width_clear_expected = furst.gratings.width_clear_delivered["ID01"]
    assert width_clear.x == width_clear_expected.x
    assert width_clear.y == width_clear_expected.y
    assert width_clear.x > design.grating.width_clear.x
    assert width_clear.y > design.grating.width_clear.y
    width_mech = result.grating.width_mech
    width_mech_expected = furst.gratings.width_mech_delivered["ID01"]
    assert width_mech.x == width_mech_expected.x
    assert width_mech.y == width_mech_expected.y

    # its grooves are those Zeiss simulated for it, at every channel's angle
    rulings = result.grating.rulings
    assert isinstance(rulings, optika.rulings.MeasuredRulings)
    assert rulings.axis_angle is not None
    assert rulings.spacing == design.grating.rulings.spacing

    # the coatings and the filter are the ones measured on the flight
    # hardware, on substrates of the thicknesses of the design
    feed = result.feed_optic.material
    assert isinstance(feed, optika.materials.MeasuredMirror)
    assert feed.substrate.thickness == design.feed_optic.material.substrate.thickness
    coating = result.grating.material
    assert isinstance(coating, optika.materials.MeasuredMirror)
    assert coating.substrate.thickness == design.grating.material.substrate.thickness
    witness = furst.filters.materials.transmission_witness_measured()
    transmission = result.filter.material.efficiency_measured.outputs
    assert np.all(transmission == witness.efficiency_measured.outputs)
    assert result.filter.thickness == furst.filters.thickness_measured.mean()

    # and the feed optic array is moved to focus it
    assert result.feed_optic.translation_focus == (
        furst.instruments.translation_focus_as_built
    )
    assert result.feed_optic.angle_focus == furst.instruments.angle_focus_as_built

    # every channel lands on the sensor
    rays = result.system.rayfunction_default
    assert np.isfinite(rays.outputs.position.x).all()
    intensity = na.nominal(rays.outputs.intensity)
    assert (intensity.mean("wavelength") > 0).all()
