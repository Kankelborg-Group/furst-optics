import astropy.units as u
import named_arrays as na
import optika
import furst_optics

def design():

    rowland_radius = 1 * u.m

    result = furst_optics.spectrographs.Spectrograph(
        name="furst",
        source=furst_optics.sources.SolarDisk(),
        front_aperture=furst_optics.apertures.FrontAperture(),
        feed_optic=furst_optics.feed_optics.FeedOptic(
            radius=3 * u.mm,
            aperture_subtent=1 * u.deg,
            aperture_height=10 * u.mm,
            material=optika.materials.Mirror(),
            rowland_radius=rowland_radius,
            # rowland_azimuth=na.linspace(10, 15, axis="angle", num=2) * u.deg,
            rowland_azimuth=10 * u.deg,
        ),
        grating=furst_optics.gratings.Grating(
            sag=optika.sags.SphericalSag(
                radius=-2 * rowland_radius,
            ),
            width_clear=100 * u.mm,
            material=optika.materials.Mirror(),
            rulings=optika.rulings.Rulings(
                spacing=1 * u.um,
                diffraction_order=1,
            ),
            rowland_radius=rowland_radius,
            rowland_azimuth=175 * u.deg,
        ),
        detector=furst_optics.detectors.Detector(
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray(
                x="detector_x",
                y="detector_y",
            ),
            num_pixel=na.Cartesian2dVectorArray(
                x=2048,
                y=1024,
            ),
            rowland_radius=rowland_radius,
            rowland_azimuth=5 * u.deg,
        ),
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(-1, 1, axis="wavelength", num=1, centers=True),
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("fx", "fy"),
                num=1,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("px", "py"),
                num=1,
                centers=True,
            )
        )
    )

    return result
