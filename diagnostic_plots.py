from astropy.io import fits
import numpy as np
from path import Path
from astropy.wcs import WCS
import warnings
import argparse
import matplotlib as mpl
from astropy.time import Time
import matplotlib.pyplot as plt
from utilities import red_gain, blue_gain, plate_scale
from astropy.coordinates import get_body, AltAz, EarthLocation
import ephem


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True


def get_camera_gain(files):
    file = files[0]
    if '-BLUE-' in file:
        return blue_gain
    elif '-RED-' in file:
        return red_gain
    else:
        raise ValueError('Could not determine camera colour from file name.')


def read_headers(input_dir_1, input_dir_2):
    files = input_dir_1.files("*.fits") + input_dir_2.files("*.fits")
    files.sort(key=lambda x: x.basename())
    headers = [fits.getheader(file) for file in files]
    return headers


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
            alts.append(None)
            azs.append(None)
    return np.array(alts), np.array(azs)


def extract_sky_background(headers, gain):
    sky_backgrounds = []
    for header in headers:
        try:
            sky_backgrounds.append(header['BACK-LVL'] / header['EXPTIME'] * gain)
        except KeyError:
            sky_backgrounds.append(None)
    return np.array(sky_backgrounds)


def extract_zp(headers):
    zps = []
    for header in headers:
        try:
            zps.append(header['ZP_10R'])
        except KeyError:
            zps.append(np.nan)
    return np.array(zps)


def plot_zp_vs_time(times, zps, output_path):
    fig, ax = plt.subplots()
    ax.plot(times, zps)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Mean ZP mag (1 e$^-$s$^{-1}$)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    plt.savefig(output_path / 'zp_vs_time.png')
    plt.close()


def plot_airmass_vs_time(times, alts, output_path):
    fig, ax = plt.subplots()
    zenith_angle = 90 - alts
    airmass = 1 / np.cos(np.radians(zenith_angle))
    ax.plot(times, airmass)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Airmass')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    plt.savefig(output_path / 'airmass_vs_time.png')
    plt.close()


def plot_hfd_vs_time(times, hfds, output_path):
    fig, ax = plt.subplots()
    ax.plot(times, hfds)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('HFD (arcsec)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    plt.savefig(output_path / 'hfd_vs_time.png')
    plt.close()


def plot_zp_vs_hfd(zps, hfds, output_path):
    fig, ax = plt.subplots()
    ax.plot(hfds, zps, 'o')
    ax.set_xlabel('HFD (arcsec)')
    ax.set_ylabel('Mean ZP mag (1 e$^-$s$^{-1}$)')
    ax.grid()
    plt.savefig(output_path / 'zp_vs_hfd.png')
    plt.close()


def plot_sky_background_vs_time(times, sky_backgrounds, output_path):
    fig, ax = plt.subplots()
    ax.plot(times, sky_backgrounds)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Sky background (e$^-$px$^{-1}$s$^{-1}$)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    plt.savefig(output_path / 'sky_background_vs_time.png')
    plt.close()


def plot_alt_az(alts, azs, times, obs_site_ephem, obs_site, output_path):
    moon = ephem.Moon()
    moon.compute(obs_site_ephem)
    illumination = moon.moon_phase * 100
    frame = AltAz(obstime=Time(times), location=obs_site)

    moon_alt_az = get_body('moon', Time(times), obs_site).transform_to(frame)
    moon_alt = moon_alt_az.alt.deg
    moon_az = moon_alt_az.az.deg

    plt.figure(figsize=(10,10))
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_ylim(90, 0)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_rgrids([0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                  labels=[r'0°', '', r'20°', '', r'40°', '', r'60°', '', r'80°', ''])
    ax.plot(azs * np.pi / 180, alts, '--', label='Telescope')
    ax.plot(moon_az * np.pi / 180, moon_alt, '--', color="gray", label=f'Moon ({illumination:.0f}\%)')
    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.1))
    plt.tight_layout()
    plt.savefig(output_path / 'alt_az.png')
    plt.close()


def plot_tracking_error(wcs, times, output_path):
    # check to see if wcs list only contains None
    if all([w is None for w in wcs]):
        return

    times = [t for t, w in zip(times, wcs) if w is not None]
    wcs = [w for w in wcs if w is not None]

    # get crval of first wcs
    initial_ra, initial_dec = wcs[0].wcs.crval

    # transform all wcs to pixel coordinates
    pixel_coords = [w.all_world2pix(initial_ra, initial_dec, 0) for w in wcs]

    # convert to 2d array
    pixel_coords = np.array(pixel_coords)
    pixel_coords -= pixel_coords[0]

    x = pixel_coords[:, 0] * plate_scale
    y = pixel_coords[:, 1] * plate_scale

    # reject more than 50 arcsec from start in any direction
    y = y[abs(x < 50)]
    x = x[abs(x < 50)]
    x = x[abs(y < 50)]
    y = y[abs(y < 50)]

    fig, ax = plt.subplots()
    delta_times = np.array([(t - times[0]).total_seconds() for t in times]) / 60
    ax.scatter(x, y, c=delta_times)
    ax.set_xlabel('X offset (arcsec)')
    ax.set_ylabel('Y offset (arcsec)')
    cbar = plt.colorbar(ax.scatter(x, y, c=delta_times))
    cbar.set_label('Time (minutes)')
    plt.savefig(output_path / 'tracking_error.png')
    plt.close()

    # now plot each individually vs time
    fig, ax = plt.subplots()
    ax.plot(times, x, label='X')
    ax.plot(times, y, label='Y')
    ax.legend()
    ax.set_ylabel('Offset (arcsec)')
    ax.set_xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.grid()
    plt.savefig(output_path / 'tracking_error_vs_time.png')
    plt.close()


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Analyse background-subtracted images')
    parser.add_argument('input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('reject_dir', type=str, help='Directory containing rejected images.')
    parser.add_argument('output_dir', type=str, help='Directory to place the diagnostic plots.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    reject_dir = Path(args.reject_dir)
    output_path = Path(args.output_dir)

    headers = read_headers(input_dir, reject_dir)
    if len(headers) == 0:
        print("No headers found in input directories.")
        return

    wcs = read_wcs(headers)
    times = extract_times(headers)
    hfds = extract_hfd(headers)
    alt, az = extract_alt_az(headers)
    gain = get_camera_gain(input_dir.files('*.fits') + reject_dir.files('*.fits'))
    sky_backgrounds = extract_sky_background(headers, gain)
    zps = extract_zp(headers) + 2.5 * np.log10(gain)

    obs_site_ephem = ephem.Observer()
    obs_site_ephem.lat = headers[0]['SITELAT']
    obs_site_ephem.lon = headers[0]['SITELONG']
    obs_site_ephem.elevation = headers[0]['SITEELEV']
    obs_site_ephem.date = ephem.Date(times[0])

    obs_site = EarthLocation(lat=obs_site_ephem.lat, lon=obs_site_ephem.lon, height=obs_site_ephem.elevation)

    print(f"Generating diagnostic plots in {output_path}")

    plot_tracking_error(wcs, times, output_path)
    plot_hfd_vs_time(times, hfds, output_path)
    plot_alt_az(alt, az, times, obs_site_ephem, obs_site, output_path)
    plot_sky_background_vs_time(times, sky_backgrounds, output_path)
    plot_zp_vs_time(times, zps, output_path)
    plot_airmass_vs_time(times, alt, output_path)
    plot_zp_vs_hfd(zps, hfds, output_path)


if __name__ == '__main__':
    main()