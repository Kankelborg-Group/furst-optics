import pytest
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins
import furst


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        furst.instruments.design(
            num_field=3,
            num_pupil=3,
        ),
    ],
)
class TestInstrument(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
):
    def test_angle_grating_input(self, a: furst.instruments.Instrument):
        result = a.angle_grating_input
        assert na.unit_normalized(result).is_equivalent(u.deg)

    @pytest.mark.parametrize("axis", ["_dummy"])
    def test_angle_grating_output(self, a: furst.instruments.Instrument, axis: str):
        result = a.angle_grating_output(axis)
        assert na.unit_normalized(result).is_equivalent(u.deg)
        assert axis in result.shape

    def test_wavelength_min(self, a: furst.instruments.Instrument):
        result = a.wavelength_min
        assert na.unit_normalized(result).is_equivalent(u.AA)
        assert (result > 0 * u.AA).all()

    def test_wavelength_max(self, a: furst.instruments.Instrument):
        result = a.wavelength_max
        assert na.unit_normalized(result).is_equivalent(u.AA)
        assert (result > a.wavelength_min).all()

    def test_wavelength_physical(self, a: furst.instruments.Instrument):
        result = a.wavelength_physical
        assert na.unit_normalized(result).is_equivalent(u.AA)
        assert (result >= a.wavelength_min).all()
        assert (result <= a.wavelength_max).all()

    def test_system(self, a: furst.instruments.Instrument):
        result = a.system
        assert isinstance(result, optika.systems.SequentialSystem)
