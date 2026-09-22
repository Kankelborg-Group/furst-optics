import dataclasses
import numpy as np
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


def _moved(instrument, **kwargs):
    """A copy of the instrument with its feed optic array moved."""
    return dataclasses.replace(
        instrument,
        feed_optic=dataclasses.replace(instrument.feed_optic, **kwargs),
    )


def test_wavelength_focus():
    """
    The wavelengths the instrument was focused at are the visible
    calibration lines scaled by the ratio of the two ruling densities, and
    they lie at the two ends of the bandpass.
    """
    first = furst.instruments.wavelength_focus_first
    last = furst.instruments.wavelength_focus_last
    assert u.isclose(first, 118.45 * u.nm, atol=0.01 * u.nm)
    assert u.isclose(last, 182.18 * u.nm, atol=0.01 * u.nm)

    instrument = furst.instruments.design(num_field=3, num_pupil=3)
    axis = instrument.feed_optic.axis_channel
    assert instrument.wavelength_min[{axis: 0}] < first
    assert first < instrument.wavelength_max[{axis: 0}]
    assert instrument.wavelength_min[{axis: ~0}] < last
    assert last < instrument.wavelength_max[{axis: ~0}]


def test_width_line():
    """
    The traced line is narrow in the channel which sees the wavelength, and
    undefined in the channels which do not.
    """
    instrument = furst.instruments.design()
    axis = instrument.feed_optic.axis_channel

    width = instrument.width_line(
        wavelength=furst.instruments.wavelength_focus_first,
        num_pupil=5,
    )
    assert width.shape == {axis: 7}

    width = width.to(u.um)
    assert 0 * u.um < width[{axis: 0}] < 5 * u.um
    assert np.isnan(width[{axis: ~0}])


def test_focus():
    """
    Focusing the design reproduces the positions stored in the package, and
    the line is narrower at those positions than a tenth of a millimeter to
    either side of them.
    """
    instrument = furst.instruments.design()
    result = instrument.focused()

    translation = result.feed_optic.translation_focus
    angle = result.feed_optic.angle_focus
    error = np.abs(translation - furst.instruments.translation_focus)
    assert np.all(error < 1e-3 * u.mm)
    error = np.abs(angle - furst.instruments.angle_focus)
    assert np.all(error < 1e-4 * u.deg)

    # the first channel is focused by the displacement of the array
    axis = "position"
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.mm, axes=axis)
    moved = _moved(result, translation_focus=translation + offset)
    width = moved.width_line(
        wavelength=furst.instruments.wavelength_focus_first,
        num_pupil=5,
    )
    width = width[{result.feed_optic.axis_channel: 0}]
    assert width[{axis: 1}] < width[{axis: 0}]
    assert width[{axis: 1}] < width[{axis: 2}]

    # and the last channel by the rotation of the array
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.deg, axes=axis)
    moved = _moved(result, angle_focus=angle + offset)
    width = moved.width_line(
        wavelength=furst.instruments.wavelength_focus_last,
        num_pupil=5,
    )
    width = width[{result.feed_optic.axis_channel: ~0}]
    assert width[{axis: 1}] < width[{axis: 0}]
    assert width[{axis: 1}] < width[{axis: 2}]


def test_focus_improves_the_design():
    """
    The design is focused, and moving the feed optic array back to where it
    would sit without the visible-blind filter makes every channel worse.
    """
    instrument = furst.instruments.design()
    assert instrument.feed_optic.translation_focus != 0 * u.mm

    wavelength = na.linspace(
        start=furst.instruments.wavelength_focus_first,
        stop=furst.instruments.wavelength_focus_last,
        axis=instrument.feed_optic.axis_channel,
        num=7,
    )

    width = instrument.width_line(wavelength, num_pupil=5)
    unfocused = _moved(
        instrument,
        translation_focus=0 * u.mm,
        angle_focus=0 * u.deg,
    )
    width_unfocused = unfocused.width_line(wavelength, num_pupil=5)

    assert np.all(width < 5 * u.um)
    assert np.all(width < width_unfocused)
