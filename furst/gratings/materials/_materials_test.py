import astropy.units as u
import named_arrays as na
import optika
import furst


def test_coating_measured():
    r = furst.gratings.materials.coating_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_coating_measured_bandpass():
    """The measurement covers the whole FURST bandpass."""
    measurement = furst.gratings.materials.coating_measured().efficiency_measured
    wavelength = measurement.inputs.wavelength
    assert wavelength.min() <= 120 * u.nm
    assert wavelength.max() >= 185 * u.nm
    assert (measurement.outputs > 0).all()
    assert (measurement.outputs < 1).all()


def test_coating_measured_efficiency():
    """
    The interpolated reflectance reproduces the plot Zeiss sent, which
    peaks near 141 nm and settles near 89 percent by 180 nm.
    """
    coating = furst.gratings.materials.coating_measured()
    wavelength = na.ScalarArray([121.6, 141, 180] * u.nm, axes="wavelength")
    r = coating.efficiency(
        rays=optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        ),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )
    assert 0.68 < r[dict(wavelength=0)] < 0.72
    assert 0.97 < r[dict(wavelength=1)] < 0.99
    assert 0.87 < r[dict(wavelength=2)] < 0.90
