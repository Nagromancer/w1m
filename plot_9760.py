from path import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import wotan
from astropy.coordinates import SkyCoord


def get_light_curve(phot, target_id, aperture_radius):
    target_phot = phot[phot["ID"] == target_id]
    time = np.array(target_phot["BJD"])
    mag = np.array(target_phot[f"MAG_{aperture_radius}"])
    mag_err = np.array(target_phot[f"MAG_ERR_{aperture_radius}"])
    return time, mag, mag_err


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32

bin_size = 5 / 1440  # 10 minutes in days
dates = ["20250823", "20250824"]

aperture_radius = 15
targ_id = 2212797391765189760

total_binned_times = []
total_binned_mags = []
total_binned_mag_errs = []
total_times = []
total_mags = []
total_mag_errs = []
for date in dates:
    try:
        target = f"Gaia DR3 {targ_id}"
        phot_path = Path(
            f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{target}/{date}-{target}-phot.fits")
        phot_table = Table.read(phot_path)
    except FileNotFoundError:
        target = target.replace(" ", "_")
        phot_path = Path(
            f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{target}/{date}-{target}-phot.fits")
        phot_table = Table.read(phot_path)

    times, mag, mag_err = get_light_curve(phot_table, targ_id, aperture_radius)

    # bin the light curve
    if bin_size is not None:
        binned_times = []
        binned_mag = []
        binned_mag_errs = []
        binned_trend = []
        for i in range(int(np.ceil(times[0] / bin_size)), int(np.ceil(times[-1] / bin_size))):
            mask = (times > i * bin_size) & (times < (i + 1) * bin_size)
            if np.sum(mask) == 0:
                continue
            binned_times.append(np.mean(times[mask]))
            binned_mag.append(np.median(mag[mask]))
            binned_mag_errs.append(np.sqrt(np.sum(mag_err[mask] ** 2)) / np.sum(mask))
    else:
        binned_times = times
        binned_mag = mag
        binned_mag_errs = mag_err

    binned_times = np.array(binned_times)
    binned_mag = np.array(binned_mag)
    binned_mag_errs = np.array(binned_mag_errs)

    total_binned_times += list(binned_times)
    total_binned_mags += list(binned_mag)
    total_binned_mag_errs += list(binned_mag_errs)
    total_times += list(times)
    total_mags += list(mag)
    total_mag_errs += list(mag_err)

total_times = np.array(total_times) - 2460900  # Convert to BJD - 2460600
total_binned_times = np.array(total_binned_times) - 2460900  # Convert to BJD - 2460600
total_mags = np.array(total_mags)
total_binned_mags = np.array(total_binned_mags)
total_mag_errs = np.array(total_mag_errs)
total_binned_mag_errs = np.array(total_binned_mag_errs)

# high error mask
high_error_mask = total_binned_mag_errs < 0.1
total_binned_times = total_binned_times[high_error_mask]
total_binned_mags = total_binned_mags[high_error_mask]
total_binned_mag_errs = total_binned_mag_errs[high_error_mask]

high_error_mask = (19.75 > total_mags) & (total_mags > 18.7)
total_times = total_times[high_error_mask]
total_mags = total_mags[high_error_mask]
total_mag_errs = total_mag_errs[high_error_mask]

plt.errorbar(total_binned_times, total_binned_mags, total_binned_mag_errs, fmt="o", markersize=2, color="black", label="Binned Data", capsize=2)
plt.errorbar(total_times, total_mags, total_mag_errs, fmt="o", markersize=2, color="black", label="Binned Data", capsize=2, alpha=0.05)
plt.gca().invert_yaxis()
# plt.ylim(19.5, 18.8)
plt.xlabel("Time (BJD - 2460900)")
plt.ylabel("$G_\mathrm{BP}$ (mag)")
plt.title(f"WDJ223900.68+665500.58")
plt.grid()
plt.show()