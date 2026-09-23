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


def _band(instrument, num=5):
    """The wavelengths each channel is focused over, in normalized coordinates."""
    return na.linspace(
        start=instrument.wavelength.min(),
        stop=instrument.wavelength.max(),
        axis="wavelength",
        num=num,
    )


def _width(instrument, wavelength, channel, num_field=5, num_pupil=5):
    """The width of the lines of one channel, combined in quadrature."""
    width = instrument.width_line(wavelength, num_field=num_field, num_pupil=num_pupil)
    width = width[{instrument.feed_optic.axis_channel: channel}]
    axis = tuple(na.shape(wavelength))
    if axis:
        width = np.sqrt(np.square(width).mean(axis))
    return width


def test_width_line():
    """
    The traced line is narrow in the channel which sees the wavelength, and
    undefined in the channels which do not.
    """
    instrument = furst.instruments.design()
    axis = instrument.feed_optic.axis_channel

    # hydrogen Lyman alpha, which only the first channel sees
    width = instrument.width_line(121.6 * u.nm, num_field=5, num_pupil=5)
    assert width.shape == {axis: 7}

    # narrower than a pixel, but never narrower than the pixel itself
    assert np.sqrt(1 / 12) * u.pix < width[{axis: 0}] < 1 * u.pix
    assert np.isnan(width[{axis: ~0}])


def test_width_line_over_a_band():
    """
    Normalized wavelengths give every channel its own range, so every
    channel sees all of them.
    """
    instrument = furst.instruments.design()
    axis = instrument.feed_optic.axis_channel
    wavelength = _band(instrument)

    width = instrument.width_line(wavelength, num_field=5, num_pupil=5)
    assert width.shape == {axis: 7, "wavelength": 5}
    assert np.all(np.isfinite(width))

    # the wavelengths span each channel without running off the detector
    physical = dataclasses.replace(
        instrument,
        wavelength=wavelength,
    ).wavelength_physical
    for channel in [0, ~0]:
        index = {axis: channel}
        assert instrument.wavelength_min[index] < physical[index].min()
        assert physical[index].max() < instrument.wavelength_max[index]


def test_focused():
    """
    Focusing the design reproduces the positions stored in the package, and
    each stage is at the minimum of its own focus curve.
    """
    instrument = furst.instruments.design()
    result = instrument.focused()

    translation = result.feed_optic.translation_focus
    angle = result.feed_optic.angle_focus
    error = np.abs(translation - furst.instruments.translation_focus)
    assert np.all(error < 1e-3 * u.mm)
    error = np.abs(angle - furst.instruments.angle_focus)
    assert np.all(error < 1e-4 * u.deg)

    wavelength = _band(result)
    axis = "position"

    # the first channel is focused by the displacement of the array
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.mm, axes=axis)
    moved = _moved(result, translation_focus=translation + offset)
    width = _width(moved, wavelength, channel=0)
    assert width[{axis: 1}] < width[{axis: 0}]
    assert width[{axis: 1}] < width[{axis: 2}]

    # and the last channel by the rotation of the array
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.deg, axes=axis)
    moved = _moved(result, angle_focus=angle + offset)
    width = _width(moved, wavelength, channel=~0)
    assert width[{axis: 1}] < width[{axis: 0}]
    assert width[{axis: 1}] < width[{axis: 2}]


def test_focused_improves_every_channel():
    """
    The design is focused, and moving the feed optic array back to where it
    would sit without the visible-blind filter makes every channel worse.
    """
    instrument = furst.instruments.design()
    assert instrument.feed_optic.translation_focus != 0 * u.mm

    unfocused = _moved(
        instrument,
        translation_focus=0 * u.mm,
        angle_focus=0 * u.deg,
    )
    wavelength = _band(instrument)

    for channel in range(7):
        width = _width(instrument, wavelength, channel)
        assert width < 1 * u.pix
        assert width < _width(unfocused, wavelength, channel)
