import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import sunpy.sun.constants
import optika
import furst

# defined in the package __init__ so that Sphinx documents them
from . import translation_focus, angle_focus


def _twist(
    feed_optic: furst.feed_optics.FeedOptic,
    grating: furst.gratings.Grating,
) -> u.Quantity | na.AbstractScalar:
    """
    The rotation of the feed optic which aims its reflected beam at the
    grating.

    The rotation is about the center of curvature of the feed optic, so it
    slides the clear aperture around the curved surface until the surface
    normal there bisects the incoming beam and the direction to the grating.
    The twist is the rotation about the vertical axis which carries the
    normal of the untwisted aperture onto that bisector, with the incoming
    beam taken to travel along the axis of the instrument.

    Parameters
    ----------
    feed_optic
        The feed optic to aim, with no twist applied yet.
    grating
        The grating to aim it at.
    """
    origin = na.Cartesian3dVectorArray() * u.mm

    # the direction from the feed optic to the grating
    position_feed = feed_optic.transformation(origin)
    position_grating = grating.transformation(origin)
    direction_out = position_grating - position_feed
    direction_out = direction_out / direction_out.length

    # the normal which reflects the incoming beam into that direction
    direction_in = na.Cartesian3dVectorArray(0, 0, 1)
    normal = direction_out - direction_in
    normal = normal / normal.length

    # the normal of the untwisted aperture, in the same coordinates
    normal_0 = feed_optic.surface.sag.normal(origin)
    normal_0 = feed_optic.transformation.transformation_linear(normal_0)

    # The rotation about the vertical axis from one normal to the other.
    # A positive rotation about the vertical axis turns the normal from the
    # axis of the instrument toward negative horizontal, so the bearing
    # measured toward positive horizontal is negated.
    angle = np.arctan2(normal.x, -normal.z) - np.arctan2(normal_0.x, -normal_0.z)
    return -angle.to(u.deg)


def _angle_from_normal(
    position: na.AbstractCartesian3dVectorArray,
    grating: furst.gratings.Grating,
) -> u.Quantity | na.AbstractScalar:
    """
    The angle between the normal of the grating and the direction to a
    point, as seen from the center of the grating.

    Parameters
    ----------
    position
        The point, in global coordinates.
    grating
        The grating.
    """
    position = grating.transformation.inverse(position)
    return np.arctan2(position.xy.length, np.abs(position.z))


def _angle_beam(
    grating: furst.gratings.Grating,
    sensor: furst.sensors.Sensor,
) -> u.Quantity | na.AbstractScalar:
    """
    The angle between the normal of the sensor and the beam from the
    grating, about the vertical axis.

    This is the yaw which turns a window mounted in front of the sensor
    to face the grating.

    Parameters
    ----------
    grating
        The grating, placed on the Rowland circle.
    sensor
        The sensor, placed on the Rowland circle.
    """
    origin = na.Cartesian3dVectorArray() * u.mm
    direction = sensor.transformation(origin) - grating.transformation(origin)
    direction = sensor.transformation.transformation_linear.inverse(direction)
    return np.arctan2(direction.x, direction.z).to(u.deg)


def _aperture_height(
    feed_optic: furst.feed_optics.FeedOptic,
    grating: furst.gratings.Grating,
    sensor: furst.sensors.Sensor,
    radius_sun: u.Quantity | na.AbstractScalar,
) -> u.Quantity | na.AbstractScalar:
    """
    The clear height of the feed optic, from the original design.

    It is the height of the sensor, grown by the astigmatism of the
    grating, which stretches the image of the Sun along the height of the
    sensor by an amount depending on the angles of incidence and
    diffraction, plus a margin.

    Parameters
    ----------
    feed_optic
        The feed optic, placed on the Rowland circle.
    grating
        The grating, placed on the Rowland circle.
    sensor
        The sensor, placed on the Rowland circle.
    radius_sun
        The angular radius of the Sun.
    """
    origin = na.Cartesian3dVectorArray() * u.mm
    alpha = _angle_from_normal(feed_optic.transformation_image(origin), grating)
    beta = _angle_from_normal(sensor.transformation(origin), grating)

    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    astigmatism = np.square(sin_alpha) * cos_beta + np.square(sin_beta) * cos_alpha
    astigmatism = astigmatism / (cos_beta * cos_alpha - np.square(sin_beta))
    astigmatism = astigmatism * np.abs(grating.sag.radius) * radius_sun.to_value(u.rad)
    astigmatism = astigmatism.max()

    height_sensor = (sensor.num_pixel_active.y * sensor.width_pixel).to(u.mm)
    margin = 2 * u.mm

    return (height_sensor + 2 * (astigmatism + margin)).to(u.mm)


def design_proposed(
    num_wavelength: int = 3,
    num_field: int = 10,
    num_pupil: int = 10,
) -> "furst.instruments.Instrument":
    """
    The FURST optical design proposed by Courrier, Kankelborg, and
    Kobayashi (2019), `cck_2019` in the original design study.

    Every parameter of the original design is reproduced here, expressed in
    a global coordinate system centered on the Rowland circle.
    The one exception is the sensor, which is the Teledyne CCD230-42 model
    from :mod:`msfc_ccd` (2048 x 1032 active pixels) in place of the
    2048 x 1040 pixels assumed by the original.

    Parameters
    ----------
    num_wavelength
        The number of wavelengths to sample in each channel.
    num_field
        The number of samples along each axis of the field of view.
    num_pupil
        The number of samples along each axis of the pupil.

    Note that this design is not focused: the visible-blind filter in front
    of the sensor moves the focus behind it, and the instrument was only
    focused once it had been assembled.
    See :func:`design`, which is.

    See Also
    --------
    :func:`design`: The final design, adjusted for the grating as delivered.
    """

    num_channels = 7
    f_ratio = 7.5
    axis_channel = "channel"

    # The original design was laid out on a breadboard, with its
    # coordinate system centered on the breadboard and the Rowland circle
    # offset from it. Here the coordinate system is centered on the Rowland
    # circle instead, so everything referenced to the breadboard is shifted
    # by the opposite of that offset.
    length_breadboard = (95 * u.imperial.inch).to(u.mm)
    z_breadboard = -length_breadboard / 2
    center_rowland = na.Cartesian3dVectorArray(
        x=-7.125 * u.imperial.inch,
        y=3.176 * u.imperial.inch,
        z=2 * u.imperial.inch,
    ).to(u.mm)

    # The field of view is sized for the largest apparent radius of the Sun,
    # while the astigmatism of the grating was sized for its mean radius.
    radius_sun_max = (32 * u.arcmin + 32 * u.arcsec) / 2
    radius_sun_mean = sunpy.sun.constants.average_angular_size

    source = furst.sources.SolarDisk(
        radius=radius_sun_max,
        translation=na.Cartesian3dVectorArray(0, 0, 1) * (-100 * u.mm + z_breadboard)
        - center_rowland,
    )

    front_aperture = furst.apertures.FrontAperture(
        translation=na.Cartesian3dVectorArray(0, 0, 1) * z_breadboard - center_rowland,
    )

    # The sensor sits on the Rowland circle a little off the axis of the
    # grating, and the grating faces it from across the circle.
    rowland_azimuth_sensor = 5 * u.deg
    rowland_azimuth_grating = 180 * u.deg - rowland_azimuth_sensor

    # The size of the grating sets the size of the Rowland circle through
    # the focal ratio, and its height is the height of the sensor grown by
    # the field of view.
    width_clear_grating = 180 * u.mm
    rowland_radius = f_ratio * width_clear_grating / 2
    radius_grating = 2 * rowland_radius

    sensor = furst.sensors.Sensor(
        material=optika.sensors.materials.e2v_ccd97(
            temperature=-55 * u.deg_C,
        ),
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth_sensor,
    )
    height_sensor = (sensor.num_pixel_active.y * sensor.width_pixel).to(u.mm)

    height_clear_grating = height_sensor + 2 * radius_grating * np.tan(radius_sun_max)

    # The grating carries the coating Zeiss measured on a test piece before
    # coating it, over a fused silica substrate, and its rulings have the
    # groove efficiency Zeiss simulated.
    material_grating = furst.gratings.materials.coating_measured()
    material_grating = dataclasses.replace(
        material_grating,
        substrate=dataclasses.replace(
            material_grating.substrate,
            thickness=35 * u.mm,
        ),
    )
    spacing_rulings = 1 / (2200 / u.mm)

    grating = furst.gratings.Grating(
        sag=optika.sags.SphericalSag(
            radius=-radius_grating,
        ),
        width_clear=na.Cartesian2dVectorArray(
            x=width_clear_grating,
            y=height_clear_grating,
        ),
        width_mech=na.Cartesian2dVectorArray(
            x=190 * u.mm,
            y=60 * u.mm,
        ),
        material=material_grating,
        rulings=furst.gratings.rulings.rulings_simulated(
            spacing=spacing_rulings,
            diffraction_order=1,
        ),
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth_grating,
    )

    # Each channel is a feed optic at its own place on the Rowland circle,
    # spread over a range of angles beyond the sensor.
    rowland_azimuth_feed = na.linspace(
        start=rowland_azimuth_sensor + 11 * u.deg,
        stop=rowland_azimuth_sensor + 11 * u.deg + 15 * u.deg,
        axis=axis_channel,
        num=num_channels,
    )

    # The feed optic is a solid rod, so its substrate is as thick as its
    # radius. It carries the coating measured on the witness samples that
    # were coated alongside it.
    radius_feed = 3 * u.mm
    material_feed = furst.feed_optics.materials.coating_witness_measured()
    material_feed = dataclasses.replace(
        material_feed,
        substrate=dataclasses.replace(
            material_feed.substrate,
            thickness=radius_feed,
        ),
    )
    feed_optic = furst.feed_optics.FeedOptic(
        radius=radius_feed,
        aperture_subtent=45 * u.deg,
        material=material_feed,
        margin_polishing=radius_feed,
        margin_mounting=3 * (2 * radius_feed),
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth_feed,
    )
    feed_optic = dataclasses.replace(
        feed_optic,
        aperture_height=_aperture_height(feed_optic, grating, sensor, radius_sun_mean),
        twist=_twist(feed_optic, grating),
    )

    # The visible-blind filter is a coated magnesium fluoride window on the
    # camera head, centered on the beam from the grating and normal to it.
    # Where exactly it sits on the head is not recorded, so it is placed
    # two inches in front of the sensor, as the original mechanical
    # placeholder and the pinhole study of the filter both assumed. The
    # whole two-inch window is taken to be clear, and its thickness is the
    # mean of the two windows that were measured, since which of them flew
    # is not recorded either.
    blind_filter = furst.filters.Filter(
        material=furst.filters.materials.transmission_witness_measured(),
        thickness=furst.filters.thickness_measured.mean(),
        radius_clear=25.4 * u.mm,
        radius_mech=25.4 * u.mm,
        distance=(2 * u.imperial.inch).to(u.mm),
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth_sensor,
        yaw=_angle_beam(grating, sensor),
    )

    # The default wavelength grid stops short of the edges of the sensor by
    # a margin of pixels, so the traced spectrum lands inside the sensor
    # with room to spare.
    num_pixel_margin = 50
    inset_wavelength = 1 - 2 * num_pixel_margin / sensor.num_pixel_active.x

    result = furst.instruments.Instrument(
        name="furst",
        source=source,
        front_aperture=front_aperture,
        feed_optic=feed_optic,
        grating=grating,
        camera=furst.cameras.Camera(
            sensor=sensor,
        ),
        filter=blind_filter,
        wavelength=na.linspace(
            start=-inset_wavelength,
            stop=inset_wavelength,
            axis="wavelength",
            num=num_wavelength,
        ),
        field=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=num_field,
            centers=True,
        ),
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=num_pupil,
            centers=True,
        ),
    )

    # The window moves the focus away from the grating by 0.6 to 0.8 mm,
    # depending on wavelength through the dispersion of magnesium fluoride.
    # Nothing here compensates for that; this is the design as proposed,
    # before the instrument was built and focused.

    return result


def design(
    num_wavelength: int = 3,
    num_field: int = 10,
    num_pupil: int = 10,
) -> "furst.instruments.Instrument":
    """
    The final FURST optical design, `zeiss_r1354mm` in the original design
    study.

    This is the design of :func:`design_proposed` adjusted for the grating
    delivered by Zeiss, which has a radius of curvature of 1354 mm instead
    of the 1350 mm originally specified.
    The Rowland circle grows to match the new grating, and the feed optics,
    the grating, and the sensor each move along the circle so as to keep
    their distance from the axis of the instrument.
    The heights of the grating and the feed optic are kept from the
    proposed design, as in the original study.

    The feed optic array is moved to the position which focuses the
    instrument, :data:`translation_focus` and :data:`angle_focus`, since
    the instrument was focused after it was assembled and with the
    visible-blind filter in place.
    See :meth:`Instrument.focused`, which finds those positions.

    Parameters
    ----------
    num_wavelength
        The number of wavelengths to sample in each channel.
    num_field
        The number of samples along each axis of the field of view.
    num_pupil
        The number of samples along each axis of the pupil.

    Examples
    --------

    Plot the layout of the instrument.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import furst

        instrument = furst.instruments.design(
            num_field=3,
            num_pupil=3,
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            instrument.system.plot(
                ax=ax,
                components=("z", "x"),
                color="black",
                kwargs_rays=dict(
                    color="tab:blue",
                    linewidth=0.5,
                ),
            )
            ax.set_aspect("equal")
    """

    result = design_proposed(
        num_wavelength=num_wavelength,
        num_field=num_field,
        num_pupil=num_pupil,
    )

    radius_grating = 1354 * u.mm
    rowland_radius = radius_grating / 2
    rowland_radius_proposed = result.grating.rowland_radius

    def rowland_azimuth(azimuth_proposed: u.Quantity | na.AbstractScalar):
        """
        The azimuth on the new Rowland circle which keeps the distance
        from the axis of the instrument the same as on the proposed one.
        """
        x = rowland_radius_proposed * np.sin(azimuth_proposed)
        return np.arcsin(x / rowland_radius).to(u.deg)

    feed_optic = dataclasses.replace(
        result.feed_optic,
        twist=0 * u.deg,
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth(result.feed_optic.rowland_azimuth),
    )

    grating = dataclasses.replace(
        result.grating,
        sag=dataclasses.replace(
            result.grating.sag,
            radius=-radius_grating,
        ),
        rowland_radius=rowland_radius,
        rowland_azimuth=180 * u.deg - rowland_azimuth(result.grating.rowland_azimuth),
    )

    sensor = dataclasses.replace(
        result.camera.sensor,
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth(result.camera.sensor.rowland_azimuth),
    )

    blind_filter = dataclasses.replace(
        result.filter,
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth(result.filter.rowland_azimuth),
        yaw=_angle_beam(grating, sensor),
    )

    # Once assembled, the instrument was focused by moving the feed optic
    # array, which is how the focus shift of the visible-blind filter was
    # taken out. These are the positions that `Instrument.focused` finds
    # for this design. The feed optics were bonded in their mounts before the array
    # was moved, so their twist is not recomputed here.
    feed_optic = dataclasses.replace(
        feed_optic,
        twist=_twist(feed_optic, grating),
        translation_focus=translation_focus,
        angle_focus=angle_focus,
    )

    return dataclasses.replace(
        result,
        feed_optic=feed_optic,
        grating=grating,
        camera=dataclasses.replace(
            result.camera,
            sensor=sensor,
        ),
        filter=blind_filter,
    )
