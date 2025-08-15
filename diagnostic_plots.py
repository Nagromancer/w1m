from astropy.wcs import WCS
import warnings
import argparse
import matplotlib as mpl
from astropy.time import Time
from utilities import red_gain, blue_gain, plate_scale
from astropy.coordinates import get_body, AltAz, EarthLocation
import ephem
import numpy as np
import imageio
import matplotlib.pyplot as plt
from astropy.io import fits
from path import Path
from PIL import Image, ImageOps
from io import BytesIO
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32


def get_camera_gain(files):
    file = files[0]
    if '-BLUE-' in file:
        return blue_gain, 'blue'
    elif '-RED-' in file:
        return red_gain, 'red'
    else:
        raise ValueError('Could not determine camera colour from file name.')


def read_headers(input_dir_1, input_dir_2):
    files = input_dir_1.files("*.fits") + input_dir_2.files("*.fits")
    files.sort(key=lambda x: x.basename())
    headers = [fits.getheader(file) for file in files]
    return headers


def get_images(input_dir_1, input_dir_2):
    files = input_dir_1.files("*.fits") + input_dir_2.files("*.fits")
    files.sort(key=lambda x: x.basename())
    return files


def read_wcs(headers):
    wcs = []
    for header in headers:
        try:
            wcs.append(WCS(header))
        except AttributeError:
            wcs.append(None)
    return wcs


def extract_times(headers):
    times = []
    for header in headers:
        times.append(Time(header['DATE-OBS'].replace('T', ' ')).datetime)
    return np.array(times)


def extract_hfd(headers):
    hfds = []
    for header in headers:
        try:
            hfds.append(header['HFD'])
        except KeyError:
            hfds.append(np.nan)
    return np.array(hfds)


def extract_alt_az(headers):
    alts = []
    azs = []
    for header in headers:
        try:
            alts.append(header['ALTITUDE'])
            azs.append(header['AZIMUTH'])
        except KeyError:
            alts.append(np.nan)
            azs.append(np.nan)
    return np.array(alts), np.array(azs)


def extract_sky_background(headers, gain, plate_scale, binning):
    sky_backgrounds = []
    for header in headers:
        try:
            sky_backgrounds.append(header['BACK-LVL'] / header['EXPTIME'] * gain / ((plate_scale * binning) ** 2))
        except KeyError:
            sky_backgrounds.append(np.nan)
    return np.array(sky_backgrounds)


def extract_zp(headers):
    zps = []
    for header in headers:
        try:
            zps.append(header['ZP_20R_e-'])
        except KeyError:
            zps.append(np.nan)
    return np.array(zps)


def extract_wind_speed(headers):
    wind_speeds = []
    median_winds = []
    wind_gusts = []
    for header in headers:
        try:
            wind_speeds.append(header['WINDSPD'])
            median_winds.append(header['MEDWIND'])
            wind_gusts.append(header['WINDGUST'])
        except KeyError:
            wind_speeds.append(np.nan)
            median_winds.append(np.nan)
            wind_gusts.append(np.nan)

    return np.array(wind_speeds), np.array(median_winds), np.array(wind_gusts)


def plot_wind_speed_vs_time(times, wind_speeds, median_winds, wind_gusts, output_path, date, target):
    fig, ax = plt.subplots()
    ax.plot(times, wind_speeds, label='Wind speed')
    ax.plot(times, median_winds, label='Median wind speed')
    ax.plot(times, wind_gusts, label='Wind gust')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Wind speed (kph)')
    ax.set_title(f"{target.replace('_', ' ')} - {date}")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.legend()
    ax.grid()
    plt.savefig(output_path / 'wind_speed_vs_time.png')
    plt.close()


def plot_zp_vs_time(times, zps, output_path, camera, date, target):
    fig, ax = plt.subplots()
    ax.plot(times, zps, color=camera)
    ax.set_xlabel('Time (UTC)')
    if camera == 'blue':
        ax.set_ylabel('Mean ${G}_\mathrm{BP}$ ZP mag (1 e$^-$s$^{-1}$)')
    else:
        ax.set_ylabel('Mean ${G}_\mathrm{RP}$ ZP mag (1 e$^-$s$^{-1}$)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    ax.grid()
    plt.savefig(output_path / 'zp_vs_time.png')
    plt.close()


def plot_airmass_vs_time(times, alts, output_path, date, target):
    fig, ax = plt.subplots()
    zenith_angle = 90 - alts
    airmass = 1 / np.cos(np.radians(zenith_angle))
    ax.plot(times, airmass, color='black')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Airmass')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.set_title(f"{target.replace('_', ' ')} - {date}")
    ax.grid()
    plt.savefig(output_path / 'airmass_vs_time.png')
    plt.close()


def plot_hfd_vs_time(times, hfds, output_path, camera, date, target):
    fig, ax = plt.subplots()
    ax.plot(times, hfds, color=camera)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('HFD (arcsec)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'hfd_vs_time.png')
    plt.close()


def plot_zp_vs_hfd(zps, hfds, output_path, camera, date, target):
    fig, ax = plt.subplots()
    ax.plot(hfds, zps, 'o', color=camera)
    ax.set_xlabel('HFD (arcsec)')
    if camera == 'blue':
        ax.set_ylabel('Mean ${G}_\mathrm{BP}$ ZP mag (1 e$^-$s$^{-1}$)')
    else:
        ax.set_ylabel('Mean ${G}_\mathrm{RP}$ ZP mag (1 e$^-$s$^{-1}$)')
    ax.grid()
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'zp_vs_hfd.png')
    plt.close()


def plot_zp_vs_airmass(zps, alts, output_path, camera, date, target):
    fig, ax = plt.subplots()
    zenith_angle = 90 - alts
    airmass = 1 / np.cos(np.radians(zenith_angle))
    ax.plot(airmass, zps, 'o', color=camera)
    ax.set_xlabel('Airmass')
    if camera == 'blue':
        ax.set_ylabel('Mean ${G}_\mathrm{BP}$ ZP mag (1 e$^-$s$^{-1}$)')
    else:
        ax.set_ylabel('Mean ${G}_\mathrm{RP}$ ZP mag (1 e$^-$s$^{-1}$)')
    ax.grid()
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'zp_vs_airmass.png')
    plt.close()


def plot_sky_background_vs_time(times, sky_backgrounds, output_path, camera, date, target):
    fig, ax = plt.subplots()
    ax.plot(times, sky_backgrounds, color=camera)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Sky background (e$^-$ s$^{-1}$ arcsec$^{-2}$)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'sky_background_vs_time.png')
    plt.close()


def plot_alt_az(alts, azs, times, obs_site_ephem, obs_site, output_path, date, target):
    moon = ephem.Moon()
    moon.compute(obs_site_ephem)
    illumination = moon.moon_phase * 100
    frame = AltAz(obstime=Time(times), location=obs_site)

    moon_alt_az = get_body('moon', Time(times), obs_site).transform_to(frame)
    moon_alt = moon_alt_az.alt.deg
    moon_az = moon_alt_az.az.deg

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_ylim(90, 0)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_rgrids([0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                  labels=[r'0°', '', r'20°', '', r'40°', '', r'60°', '', r'80°', ''], alpha=0.5)
    ax.plot(azs * np.pi / 180, alts, '--', label='Telescope', color="green")
    ax.plot(moon_az * np.pi / 180, moon_alt, '--', color="gray", label=f'Moon ({illumination:.0f}\%)')

    # plot the positions every hour and label them. Interpolate between the points
    for i in range(0, len(azs) - 1):
        if times[i].hour != times[i + 1].hour:
            az_interp = np.linspace(azs[i], azs[i + 1], 10)
            alt_interp = np.interp(az_interp, [azs[i], azs[i + 1]], [alts[i], alts[i + 1]])
            ax.plot(az_interp[-1] * np.pi / 180, alt_interp[-1], 'ko', markersize=2)
            ax.text(az_interp[-1] * np.pi / 180, alt_interp[-1], f"{times[i + 1].strftime('%H')}h", fontsize=20)

    # do the same for the moon
    for i in range(0, len(moon_az) - 1):
        if times[i].hour != times[i + 1].hour:
            az_interp = np.linspace(moon_az[i], moon_az[i + 1], 10)
            alt_interp = np.interp(az_interp, [moon_az[i], moon_az[i + 1]], [moon_alt[i], moon_alt[i + 1]])
            if alt_interp[-1] > 0:
                ax.plot(az_interp[-1] * np.pi / 180, alt_interp[-1], 'ko', markersize=2)
                ax.text(az_interp[-1] * np.pi / 180, alt_interp[-1], f"{times[i + 1].strftime('%H')}h", fontsize=20)

    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.05))
    ax.set_title(f"{target.replace('_', ' ')} - {date}", pad=50)

    plt.tight_layout()
    plt.savefig(output_path / 'alt_az.png')
    plt.close()


def plot_tracking_error(wcs, times, output_path, camera, date, target, binning):
    # check to see if wcs list only contains None
    if all([w is None for w in wcs]):
        return

    unvalidated_times = [t for t, w in zip(times, wcs) if w is not None]
    wcs = [w for w in wcs if w is not None]

    # get crval of first wcs
    initial_ra, initial_dec = wcs[0].wcs.crval

    # transform all wcs to pixel coordinates
    pixel_coords = []
    times = []
    for t, w in zip(unvalidated_times, wcs):
        try:
            pixel_coords.append(w.all_world2pix(initial_ra, initial_dec, 0))
        except Exception:
            # if the wcs is not valid, skip it
            continue
        times.append(t)

    # convert to 2d array
    pixel_coords = np.array(pixel_coords)
    pixel_coords -= pixel_coords[0]

    x = pixel_coords[:, 0] * plate_scale * binning
    y = pixel_coords[:, 1] * plate_scale * binning

    x = x if camera == 'blue' else -x
    y = -y

    # reject more than 50 arcsec from start in any direction. making sure x, y, t are same length
    x, y, times = zip(*[(x_, y_, t) for x_, y_, t in zip(x, y, times) if np.sqrt(x_ ** 2 + y_ ** 2) < 200])

    fig, ax = plt.subplots()
    delta_times = np.array([(t - times[0]).total_seconds() for t in times]) / 60
    ax.scatter(x, y, c=delta_times)
    ax.set_xlabel('RA offset (arcsec)')
    ax.set_ylabel('Dec offset (arcsec)')
    cbar = plt.colorbar(ax.scatter(x, y, c=delta_times))
    cbar.set_label('Time (minutes)')
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'tracking_error.png')
    plt.close()

    # now plot each individually vs time
    fig, ax = plt.subplots()
    c1 = "red" if camera == "red" else "blue"
    c2 = "purple" if camera == "red" else "green"
    ax.plot(times, x, label='RA', color=c1)
    ax.plot(times, y, label='Dec', color=c2)
    ax.legend()
    ax.set_ylabel('Offset (arcsec)')
    ax.set_xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    ax.set_title(f"{target.replace('_', ' ')} - {date} ({camera.capitalize()})")
    plt.savefig(output_path / 'tracking_error_vs_time.png')
    plt.close()


def process_fits_image(fits_file):
    """Load and process a single FITS image."""
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Assuming image data is in the primary HDU
        data = np.nan_to_num(data)  # Handle NaN values
        data = np.clip(data, np.percentile(data, 1), np.percentile(data, 99))  # Contrast stretch
    return data


def pad_image_to_size(image, target_size=(1024, 1024)):
    """Resize or pad the image to the exact target size."""
    return ImageOps.pad(image, target_size, method=Image.Resampling.BICUBIC)


def create_movie_from_fits(fits_files, output_path, fps):
    """Convert a series of FITS images into a video."""
    images = []

    print(f"Creating movie from {len(fits_files)} FITS images...")
    for fits_file in tqdm(fits_files):
        data = process_fits_image(fits_file)
        header = fits.getheader(fits_file)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(data, cmap='gray', origin='lower')
        ax.axis('off')
        ax.text(0.05, 0.05, header['DATE-OBS'].replace("T", " "), color='red', transform=ax.transAxes, fontsize=18,
                ha='left', va='top')

        # Save figure to an in-memory buffer instead of a file
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)

        # Resize image to 1024x1024 for FFmpeg
        image = pad_image_to_size(image, target_size=(1024, 1024))
        images.append(image)

    # Create video using FFmpeg format
    imageio.mimsave(output_path / "movie.mp4", images, fps=fps, format='FFMPEG')
    print(f"Movie saved as {output_path / 'movie.mp4'}")


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Analyse background-subtracted images')
    parser.add_argument('input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('reject_dir', type=str, help='Directory containing rejected images.')
    parser.add_argument('output_dir', type=str, help='Directory to place the diagnostic plots.')
    parser.add_argument('binning', type=int, help='Binning factor.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    reject_dir = Path(args.reject_dir)
    output_path = Path(args.output_dir)

    headers = read_headers(input_dir, reject_dir)
    if len(headers) == 0:
        print("No headers found in input directories.")
        return

    # get date from parent directory
    date = input_dir.parent.parent.basename()
    date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    target = input_dir.parent.basename()

    wcs = read_wcs(headers)
    times = extract_times(headers)
    hfds = extract_hfd(headers)
    alt, az = extract_alt_az(headers)
    gain, cam_colour = get_camera_gain(input_dir.files('*.fits') + reject_dir.files('*.fits'))
    sky_backgrounds = extract_sky_background(headers, gain, plate_scale, args.binning)
    zps = extract_zp(headers)
    wind_speeds, median_winds, wind_gusts = extract_wind_speed(headers)

    obs_site_ephem = ephem.Observer()
    obs_site_ephem.lat = headers[0]['SITELAT']
    obs_site_ephem.lon = headers[0]['SITELONG']
    obs_site_ephem.elevation = headers[0]['SITEELEV']
    obs_site_ephem.date = ephem.Date(times[0])

    obs_site = EarthLocation(lat=obs_site_ephem.lat, lon=obs_site_ephem.lon, height=obs_site_ephem.elevation)

    print(f"Generating diagnostic plots in {output_path}")

    plot_tracking_error(wcs, times, output_path, cam_colour, date, target, args.binning)
    plot_hfd_vs_time(times, hfds, output_path, cam_colour, date, target)
    plot_alt_az(alt, az, times, obs_site_ephem, obs_site, output_path, date, target)
    plot_sky_background_vs_time(times, sky_backgrounds, output_path, cam_colour, date, target)
    plot_zp_vs_time(times, zps, output_path, cam_colour, date, target)
    plot_airmass_vs_time(times, alt, output_path, date, target)
    plot_zp_vs_hfd(zps, hfds, output_path, cam_colour, date, target)
    plot_zp_vs_airmass(zps, alt, output_path, cam_colour, date, target)
    plot_wind_speed_vs_time(times, wind_speeds, median_winds, wind_gusts, output_path, date, target)

    # images = get_images(input_dir, reject_dir)
    # create_movie_from_fits(images, output_path, 30)
    #
    # temp_frame = Path("temp_frame.png")
    # if temp_frame.exists():
    #     temp_frame.remove()


if __name__ == '__main__':
    main()
