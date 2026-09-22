import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import furst


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

    width = furst.instruments.width_line(
        instrument=instrument,
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
    result = furst.instruments.focus(instrument)

    translation = result.feed_optic.translation_focus
    angle = result.feed_optic.angle_focus
    error = np.abs(translation - furst.instruments.translation_focus)
    assert np.all(error < 1e-3 * u.mm)
    error = np.abs(angle - furst.instruments.angle_focus)
    assert np.all(error < 1e-4 * u.deg)

    # the first channel is focused by the displacement of the array
    axis = "position"
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.mm, axes=axis)
    width = furst.instruments.width_line(
        instrument=_moved(result, translation_focus=translation + offset),
        wavelength=furst.instruments.wavelength_focus_first,
        num_pupil=5,
    )
    width = width[{result.feed_optic.axis_channel: 0}]
    assert width[{axis: 1}] < width[{axis: 0}]
    assert width[{axis: 1}] < width[{axis: 2}]

    # and the last channel by the rotation of the array
    offset = na.ScalarArray(np.array([-0.1, 0, 0.1]) * u.deg, axes=axis)
    width = furst.instruments.width_line(
        instrument=_moved(result, angle_focus=angle + offset),
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

    width = furst.instruments.width_line(instrument, wavelength, num_pupil=5)
    width_unfocused = furst.instruments.width_line(
        instrument=_moved(
            instrument,
            translation_focus=0 * u.mm,
            angle_focus=0 * u.deg,
        ),
        wavelength=wavelength,
        num_pupil=5,
    )

    assert np.all(width < 5 * u.um)
    assert np.all(width < width_unfocused)
