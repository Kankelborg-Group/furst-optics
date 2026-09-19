import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst
from furst.feed_optics import materials


def _reflectivity(
    coating: optika.materials.AbstractMultilayerMirror,
    wavelength: u.Quantity | na.AbstractScalar,
    angle: u.Quantity = None,
) -> na.AbstractScalar:
    """The reflectivity of a coating at the given wavelength and angle."""
    if angle is None:
        angle = materials.angle_witness
    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )
    return coating.efficiency(
        rays=rays,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )


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
    index = coating.layers[0].chemical.n(materials.wavelength_design)
    index = float(na.as_named_array(np.real(index)).ndarray)
    expected = materials.wavelength_design / (4 * index)
    assert u.isclose(thickness, expected.to(u.nm))


def test_coating_design_reflectivity():
    """
    The model reproduces the reflectance the vendor specifies.

    Acton quotes 78 to 83 percent at 121.6 nm at normal incidence for
    this coating, which is the only published number we can check
    against.
    """
    coating = furst.feed_optics.materials.coating_design()
    r = _reflectivity(coating, materials.wavelength_design)
    r = float(na.as_named_array(r).ndarray)
    assert 0.78 < r < 0.83


def test_coating_witness_measured():
    r = furst.feed_optics.materials.coating_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_coating_witness_measured_angle():
    """The witness samples are taken to have been measured at normal incidence."""
    r = furst.feed_optics.materials.coating_witness_measured()
    assert r.efficiency_measured.inputs.direction == materials.angle_witness


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
    """
    The fit is closer to the measurement than the design it starts from,
    across the FURST bandpass.
    """
    measurement = furst.feed_optics.materials.coating_witness_measured()
    measured = measurement.efficiency_measured
    wavelength = measured.inputs.wavelength

    band = (wavelength > 120 * u.nm) & (wavelength < 185 * u.nm)
    expected = measured.outputs[band]

    def rms(coating):
        r = _reflectivity(coating, wavelength[band])
        return np.sqrt(np.mean(np.square(r - expected)))

    rms_design = rms(furst.feed_optics.materials.coating_design())
    rms_fit = rms(furst.feed_optics.materials.coating_witness_fit())

    assert rms_fit < rms_design
    # the residual is dominated by the coating being proprietary, so this
    # is a loose bound meant only to catch the fit breaking
    assert rms_fit < 0.05
