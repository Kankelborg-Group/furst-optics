import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst
from furst.feed_optics import materials
from furst.feed_optics.materials._materials import _efficiency, _witness_data


def _rms(coating, angle):
    """The residual of a coating against the witness measurement."""
    data = _witness_data()
    residual = _efficiency(coating, data.inputs, angle) - data.outputs
    return float(na.as_named_array(np.sqrt(np.mean(np.square(residual)))).ndarray)


def test_coating_design():
    r = furst.feed_optics.materials.coating_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_coating_design_quarter_wave():
    """
    The magnesium fluoride is a quarter wave at the design wavelength.

    This is the whole content of the model, so it is worth pinning.
    """
    coating = furst.feed_optics.materials.coating_design()
    thickness = coating.layers[0].thickness
    chemical = optika.chemicals.Chemical(coating.layers[0].chemical)
    index = chemical.n(materials.wavelength_design)
    index = float(na.as_named_array(np.real(index)).ndarray)
    expected = materials.wavelength_design / (4 * index)
    assert u.isclose(thickness, expected.to(u.nm))


def test_coating_design_reflectance():
    """
    The model reproduces the reflectance the vendor specifies.

    Acton quotes 78 to 83 percent at 121.6 nm at normal incidence for
    this coating, which is the only published number we can check
    against.
    """
    coating = furst.feed_optics.materials.coating_design()
    r = _efficiency(
        coating,
        materials.wavelength_design,
        materials.angle_specification,
    )
    r = float(na.as_named_array(r).ndarray)
    assert 0.78 < r < 0.83


def test_coating_witness_measured():
    r = furst.feed_optics.materials.coating_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_coating_witness_measured_angle():
    """The measurement reports the solved angle rather than an assumed one."""
    r = furst.feed_optics.materials.coating_witness_measured()
    assert r.efficiency_measured.inputs.direction == materials.angle_witness()


def test_angle_witness():
    """
    The solved angle is steep.

    The angle was never recorded. The measured curve is only consistent
    with a steep geometry, and this pins that conclusion so that a
    change to the fit which quietly returns to near-normal incidence is
    caught.
    """
    angle = furst.feed_optics.materials.angle_witness()
    assert na.unit(angle).is_equivalent(u.deg)
    assert 50 * u.deg < angle < 85 * u.deg


def test_angle_witness_beats_normal_incidence():
    """
    The solved angle fits the measurement better than normal incidence.

    The objective has a shallow second minimum near normal incidence, so
    this guards against the fit falling into it.
    """
    coating = furst.feed_optics.materials.coating_witness_fit()
    solved = _rms(coating, furst.feed_optics.materials.angle_witness())
    normal = _rms(coating, 0 * u.deg)
    assert solved < normal / 2


def test_coating_witness_fit():
    r = furst.feed_optics.materials.coating_witness_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)


def test_coating_witness_fit_aluminum_opaque():
    """
    The fit leaves the aluminum at its opaque design thickness.

    Fitting it as well admits an unphysical semi-transparent solution.
    """
    fit = furst.feed_optics.materials.coating_witness_fit()
    design = furst.feed_optics.materials.coating_design()
    assert fit.layers[1].thickness == design.layers[1].thickness


def test_coating_witness_fit_closer_than_design():
    """The fit is closer to the measurement than the design it starts from."""
    angle = furst.feed_optics.materials.angle_witness()
    rms_fit = _rms(furst.feed_optics.materials.coating_witness_fit(), angle)
    rms_design = _rms(furst.feed_optics.materials.coating_design(), angle)

    assert rms_fit < rms_design
    # the residual is dominated by the coating being proprietary, so this
    # is a loose bound meant only to catch the fit breaking
    assert rms_fit < 0.05
