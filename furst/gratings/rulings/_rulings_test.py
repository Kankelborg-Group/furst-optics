import pytest
import numpy as np
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


def test_rulings_simulated_matches_sinusoid():
    """
    The groove efficiency recovered from the Zeiss simulation agrees with
    optika's own scalar model of a sinusoidal profile of the same depth
    across the bandpass.

    This is both a check on the decomposition and the reason for the
    requirement on optika: before optika 2.9 the sinusoidal model
    returned the Bessel function unsquared, about 0.55 here.
    """
    depth = 42 * u.nm
    rulings = furst.gratings.rulings.rulings_simulated(depth)
    sinusoid = optika.rulings.SinusoidalRulings(
        spacing=rulings.spacing,
        depth=depth,
        diffraction_order=rulings.diffraction_order,
    )

    angle = furst.gratings.rulings.angle_simulated
    rays = optika.rays.RayVectorArray(
        wavelength=rulings.efficiency_measured.inputs.wavelength,
        position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )
    normal = na.Cartesian3dVectorArray(0, 0, -1)

    grooves = rulings.efficiency(rays, normal)
    expected = sinusoid.efficiency(rays, normal)

    assert np.abs(grooves - expected).max() < 0.02


@pytest.mark.parametrize(
    argnames="serial_number",
    argvalues=furst.gratings.rulings.serial_numbers_delivered,
)
def test_efficiency_delivered(serial_number: str):
    result = furst.gratings.rulings.efficiency_delivered(serial_number)
    assert isinstance(result, na.FunctionArray)
    wavelength = result.inputs.wavelength
    assert wavelength.min() <= 121 * u.nm
    assert wavelength.max() >= 189 * u.nm
    angle = result.inputs.direction
    assert np.all(angle == furst.gratings.rulings.angles_delivered)
    assert result.outputs.shape == {"wavelength": wavelength.size, "angle": 3}
    # Zeiss found 24 to 33 percent on both gratings at every angle
    assert (result.outputs > 0.24).all()
    assert (result.outputs < 0.33).all()


def test_efficiency_delivered_serial_number_unknown():
    with pytest.raises(ValueError):
        furst.gratings.rulings.efficiency_delivered("ID04")


def test_efficiency_delivered_matches_prediction():
    """
    The flight grating came out 38.5 nm deep, and its efficiency is close
    to the one Zeiss predicted for a 39 nm profile before it was made.
    """
    angle = furst.gratings.rulings.angle_simulated
    delivered = furst.gratings.rulings.efficiency_delivered("ID01")
    index = {
        "angle": np.flatnonzero(furst.gratings.rulings.angles_delivered == angle)[0]
    }
    predicted = furst.gratings.rulings.efficiency_simulated(39 * u.nm)
    wavelength = na.linspace(121, 185, axis="wavelength", num=65) * u.nm
    result = na.interp(
        x=wavelength,
        xp=delivered.inputs.wavelength,
        fp=delivered.outputs[index],
    )
    expected = na.interp(
        x=wavelength,
        xp=predicted.inputs.wavelength,
        fp=predicted.outputs,
    )
    assert np.abs(result - expected).max() < 0.015


@pytest.mark.parametrize(
    argnames="serial_number",
    argvalues=furst.gratings.rulings.serial_numbers_delivered,
)
def test_rulings_delivered(serial_number: str):
    """
    Dividing out the coating leaves a groove efficiency that is higher than
    the simulated total at every angle.
    """
    rulings = furst.gratings.rulings.rulings_delivered(serial_number)
    assert isinstance(rulings, optika.rulings.MeasuredRulings)
    assert rulings.diffraction_order == 1
    assert u.isclose(rulings.spacing, 1 / (2200 / u.mm))
    assert rulings.axis_angle == "angle"
    assert rulings.shape == {}

    total = furst.gratings.rulings.efficiency_delivered(serial_number)
    grooves = rulings.efficiency_measured.outputs
    assert (grooves > total.outputs).all()
    assert (grooves > 0.27).all()
    assert (grooves < 0.36).all()


def test_rulings_delivered_efficiency():
    """
    The rulings reproduce the groove efficiency at the angles Zeiss
    simulated, and interpolate between them.
    """
    rulings = furst.gratings.rulings.rulings_delivered()
    measurement = rulings.efficiency_measured
    angles = furst.gratings.rulings.angles_delivered

    angle = na.ScalarArray(
        u.Quantity([angles[0], angles[:2].mean(), angles[1]]),
        axes="ray",
    )
    rays = optika.rays.RayVectorArray(
        wavelength=measurement.inputs.wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )
    result = rulings.efficiency(rays, normal=na.Cartesian3dVectorArray(0, 0, -1))
    assert result.shape == {"wavelength": measurement.inputs.wavelength.size, "ray": 3}

    first = measurement.outputs[{"angle": 0}]
    second = measurement.outputs[{"angle": 1}]
    assert np.allclose(result[{"ray": 0}], first)
    assert np.allclose(result[{"ray": 1}], (first + second) / 2)
    assert np.allclose(result[{"ray": 2}], second)
