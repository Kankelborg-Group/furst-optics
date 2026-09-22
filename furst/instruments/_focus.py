import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import furst

# defined in the package __init__ so that Sphinx documents them
from . import wavelength_focus_first, wavelength_focus_last

__all__ = [
    "width_line",
    "focus",
]


def width_line(
    instrument: "furst.instruments.Instrument",
    wavelength: u.Quantity | na.AbstractScalar,
    num_pupil: int = 11,
) -> na.AbstractScalar:
    """
    The width of the spectral line formed at the given wavelength, measured
    along the dispersion direction, for every channel of the instrument.

    This is the quantity minimized to focus the instrument, the analogue of
    the width of a calibration lamp line measured on the bench.
    It is computed from a point source at the center of the field, so it is
    the blur of the optics alone and not the disk-integrated line spread
    function.
    Channels which do not see the given wavelength catch no rays, and their
    width is :obj:`numpy.nan`.

    Parameters
    ----------
    instrument
        The instrument to trace rays through.
    wavelength
        The wavelength of the line, in physical units.
        Any axes of this array are carried through to the result, so a grid
        of wavelengths can be measured at once.
    num_pupil
        The number of samples along each axis of the pupil.

    Examples
    --------

    Plot the focus curve of the first channel, the width of a line as the
    feed optic array slides along the axis of the instrument.

    .. jupyter-execute::

        import dataclasses
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import furst

        # Load the design and the wavelength its first channel was focused at
        instrument = furst.instruments.design()
        wavelength = furst.instruments.wavelength_focus_first

        # Slide the feed optic array along the axis of the instrument
        translation = na.linspace(-1, 2, axis="position", num=13) * u.mm
        feed_optic = dataclasses.replace(
            instrument.feed_optic,
            translation_focus=translation,
        )
        instrument = dataclasses.replace(instrument, feed_optic=feed_optic)

        # Measure the width of the line in the first channel
        axis_channel = instrument.feed_optic.axis_channel
        width = furst.instruments.width_line(instrument, wavelength)
        width = width[{axis_channel: 0}].to(u.um)

        # Plot the focus curve
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(translation, width, ax=ax, marker="o");
            ax.set_xlabel(f"displacement of the array ({translation.unit:latex_inline})");
            ax.set_ylabel(f"width of the line ({width.unit:latex_inline})");
    """
    instrument = dataclasses.replace(
        instrument,
        wavelength=wavelength,
        field=na.Cartesian2dVectorArray(0, 0),
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=num_pupil,
            centers=True,
        ),
    )

    rays = instrument.system.rayfunction_default.outputs

    axis = ("pupil_x", "pupil_y")
    weight = rays.unvignetted.astype(float)
    position = rays.position.x

    # a channel which sees none of the given wavelength catches no rays,
    # and divides zero by zero to give the documented NaN
    with np.errstate(invalid="ignore"):
        mean = (position * weight).sum(axis) / weight.sum(axis)
        variance = (np.square(position - mean) * weight).sum(axis) / weight.sum(axis)
        result = np.sqrt(variance)

    return result


def _vertex(
    inputs: na.AbstractScalar,
    outputs: na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    """
    The position of the minimum of a parabola fitted to a focus curve.

    The square of the width of the line is the quantity which is quadratic
    in the defocus, so it is what is fitted, as on the bench.

    Parameters
    ----------
    inputs
        The sampled positions of the feed optic array.
    outputs
        The width of the line measured at each position.
    axis
        The logical axis along which the curve is sampled.
    """
    fit = na.PolynomialFitFunctionArray.from_degree(
        inputs=inputs,
        outputs=np.square(outputs),
        degree=2,
        axis_polynomial=axis,
        center=inputs.mean(axis),
    )
    name_linear, name_quadratic = fit.coefficient_names[1:]
    coefficients = fit.coefficients.components
    return fit.center - coefficients[name_linear] / (2 * coefficients[name_quadratic])


def focus(
    instrument: "furst.instruments.Instrument",
    wavelength_first: u.Quantity = wavelength_focus_first,
    wavelength_last: u.Quantity = wavelength_focus_last,
    translation: None | na.AbstractScalar = None,
    angle: None | na.AbstractScalar = None,
    num_pupil: int = 11,
) -> "furst.instruments.Instrument":
    """
    Focus the instrument by moving the feed optic array, reproducing the
    procedure carried out during assembly.

    The array is mounted on two stages, and each is used to focus one end of
    the spectrum:

    #. The lower stage slides the whole array along the axis of the
       instrument until the line at ``wavelength_first`` is narrowest in the
       first channel.
    #. The upper stage pivots the array about the first feed optic, which
       leaves the first channel where it is, until the line at
       ``wavelength_last`` is narrowest in the last channel.

    The remaining channels are not adjusted; they land wherever the
    mechanism puts them, as they do on the bench.

    Each step samples the width of the line over a grid of positions and
    fits a parabola to it, and the two grids are traced all at once, so the
    whole procedure costs two raytraces.

    Parameters
    ----------
    instrument
        The instrument to focus.
    wavelength_first
        The wavelength at which the first channel is focused.
    wavelength_last
        The wavelength at which the last channel is focused.
    translation
        The displacements of the array to sample in the first step.
    angle
        The rotations of the array to sample in the second step.
    num_pupil
        The number of samples along each axis of the pupil.

    Examples
    --------

    Focus the design and print the positions found.

    .. jupyter-execute::

        import furst

        instrument = furst.instruments.design()
        instrument = furst.instruments.focus(instrument)

        print(f"{instrument.feed_optic.translation_focus=:.4f}")
        print(f"{instrument.feed_optic.angle_focus=:.5f}")
    """
    axis = "position"

    if translation is None:
        translation = na.linspace(-1, 2, axis=axis, num=13) * u.mm
    if angle is None:
        angle = na.linspace(-0.3, 0.3, axis=axis, num=13) * u.deg

    feed_optic = instrument.feed_optic
    axis_channel = feed_optic.axis_channel

    def moved(translation_focus, angle_focus):
        return dataclasses.replace(
            instrument,
            feed_optic=dataclasses.replace(
                feed_optic,
                translation_focus=translation_focus,
                angle_focus=angle_focus,
            ),
        )

    # the lower stage, which focuses the first channel
    width = width_line(
        instrument=moved(translation, 0 * u.deg),
        wavelength=wavelength_first,
        num_pupil=num_pupil,
    )
    translation_focus = _vertex(
        inputs=translation,
        outputs=width[{axis_channel: 0}],
        axis=axis,
    )

    # the upper stage, which focuses the last channel
    width = width_line(
        instrument=moved(translation_focus, angle),
        wavelength=wavelength_last,
        num_pupil=num_pupil,
    )
    angle_focus = _vertex(
        inputs=angle,
        outputs=width[{axis_channel: ~0}],
        axis=axis,
    )

    return moved(translation_focus, angle_focus)
