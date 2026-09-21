import pytest
import astropy.units as u
import named_arrays as na
import optika
import furst


@pytest.mark.parametrize("depth", furst.gratings.rulings.depths_simulated)
def test_efficiency_simulated(depth: u.Quantity):
    result = furst.gratings.rulings.efficiency_simulated(depth)
    assert isinstance(result, na.FunctionArray)
    wavelength = result.inputs.wavelength
    assert wavelength.min() <= 121 * u.nm
    assert wavelength.max() >= 185 * u.nm
    assert result.inputs.direction == furst.gratings.rulings.angle_simulated
    # Zeiss found 25 to 32 percent for every depth it considered
    assert (result.outputs > 0.24).all()
    assert (result.outputs < 0.33).all()


def test_efficiency_simulated_depth_unknown():
    with pytest.raises(ValueError):
        furst.gratings.rulings.efficiency_simulated(40 * u.nm)


def test_rulings_simulated():
    result = furst.gratings.rulings.rulings_simulated()
    assert isinstance(result, optika.rulings.MeasuredRulings)
    assert result.diffraction_order == 1
    assert u.isclose(result.spacing, 1 / (2200 / u.mm))


def test_rulings_simulated_grooves_alone():
    """
    Dividing out the coating leaves a groove efficiency that is higher
    than the simulated total, since the coating reflects less than all of
    the light, and that is nearly flat across the bandpass.
    """
    rulings = furst.gratings.rulings.rulings_simulated()
    total = furst.gratings.rulings.efficiency_simulated()
    grooves = rulings.efficiency_measured.outputs
    assert (grooves > total.outputs).all()
    assert (grooves > 0.29).all()
    assert (grooves < 0.34).all()


def test_rulings_simulated_efficiency():
    """The rulings interpolate the groove efficiency in wavelength."""
    rulings = furst.gratings.rulings.rulings_simulated()
    rays = optika.rays.RayVectorArray(
        wavelength=na.ScalarArray([125, 150, 175] * u.nm, axes="wavelength"),
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )
    result = rulings.efficiency(rays, normal=na.Cartesian3dVectorArray(0, 0, -1))
    assert result.shape == {"wavelength": 3}
    assert (result > 0.29).all()
    assert (result < 0.34).all()
