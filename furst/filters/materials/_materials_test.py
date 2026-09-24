import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst
from furst.filters import materials


def test_transmission_witness_measured():
    r = furst.filters.materials.transmission_witness_measured()
    assert isinstance(r, optika.materials.MeasuredFilter)
    assert isinstance(r.medium, optika.materials.Dielectric)
    assert not r.is_mirror


def test_transmission_witness_measured_angle():
    """The witness sample is taken to have been measured at normal incidence."""
    r = furst.filters.materials.transmission_witness_measured()
    assert r.efficiency_measured.inputs.direction == materials.angle_witness


def test_transmission_witness_measured_values():
    """
    The measurement covers the bandpass, transmits 10 to 25 percent there,
    and does not double count the absorption of the window.
    """
    r = furst.filters.materials.transmission_witness_measured()
    measurement = r.efficiency_measured
    wavelength = measurement.inputs.wavelength
    assert wavelength.min() <= 120 * u.nm
    assert wavelength.max() >= 185 * u.nm

    band = (wavelength >= 120 * u.nm) & (wavelength <= 185 * u.nm)
    transmission = measurement.outputs[band]
    assert (transmission > 0.1).all()
    assert (transmission < 0.25).all()

    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )
    assert (r.attenuation(rays) == 0 / u.mm).all()
    assert (r.index_refraction(rays) > 1.3).all()


def test_transmission_design():
    r = furst.filters.materials.transmission_design()
    assert isinstance(r, optika.materials.MeasuredFilter)
    assert isinstance(r.medium, optika.materials.Dielectric)
    assert r.is_medium_measured
    assert not r.is_mirror


def test_transmission_design_values():
    """
    The published curve covers the bandpass from 121 nm, transmits 8 to 21
    percent there, and is below the witness sample measured for the flight
    filter everywhere past its steep short-wavelength edge.
    """
    r = furst.filters.materials.transmission_design()
    measurement = r.efficiency_measured
    wavelength = measurement.inputs.wavelength
    assert wavelength.min() < 122 * u.nm
    assert wavelength.max() >= 420 * u.nm

    band = wavelength <= 185 * u.nm
    transmission = measurement.outputs[band]
    assert (transmission > 0.08).all()
    assert (transmission < 0.21).all()

    witness = furst.filters.materials.transmission_witness_measured()
    rays = optika.rays.RayVectorArray(
        wavelength=na.linspace(125, 185, axis="wavelength", num=61) * u.nm,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )
    normal = na.Cartesian3dVectorArray(0, 0, -1)
    assert np.all(r.efficiency(rays, normal) < witness.efficiency(rays, normal))
