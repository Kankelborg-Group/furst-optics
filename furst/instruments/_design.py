import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import furst


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


def design():
    """
    The FURST design of Courrier, Kankelborg, and Kobayashi (2019),
    `cck_2019` of the original design study.
    """

    num_channels = 7
    f_ratio = 7.5

    # The instrument is laid out on a breadboard, which sets where the Sun
    # and the front aperture sit along the axis.
    length_breadboard = (95 * u.imperial.inch).to(u.mm)
    z_breadboard = -length_breadboard / 2

    source = furst.sources.SolarDisk(
        translation=na.Cartesian3dVectorArray(0, 0, 1) * (-100 * u.mm + z_breadboard),
    )

    front_aperture = furst.apertures.FrontAperture(
        translation=na.Cartesian3dVectorArray(0, 0, 1) * z_breadboard,
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

    height_clear_grating = height_sensor + 2 * radius_grating * np.tan(source.radius)

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
        material=optika.materials.Mirror(),
        rulings=optika.rulings.Rulings(
            spacing=1 / (2200 / u.mm),
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
        axis="channel",
        num=num_channels,
    )

    radius_feed = 3 * u.mm
    feed_optic = furst.feed_optics.FeedOptic(
        radius=radius_feed,
        aperture_subtent=45 * u.deg,
        material=optika.materials.Mirror(),
        margin_polishing=radius_feed,
        margin_mounting=3 * (2 * radius_feed),
        rowland_radius=rowland_radius,
        rowland_azimuth=rowland_azimuth_feed,
    )
    feed_optic = dataclasses.replace(
        feed_optic,
        aperture_height=_aperture_height(feed_optic, grating, sensor, source.radius),
        twist=_twist(feed_optic, grating),
    )

    result = furst.instruments.Instrument(
        name="furst",
        source=source,
        front_aperture=front_aperture,
        feed_optic=feed_optic,
        grating=grating,
        camera=furst.cameras.Camera(
            sensor=sensor,
        ),
        wavelength=na.linspace(-1, 1, axis="wavelength", num=3, centers=True),
        field=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=3,
            centers=True,
        ),
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=3,
            centers=True,
        ),
    )

    return result
