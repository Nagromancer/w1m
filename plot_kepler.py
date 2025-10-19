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
dates = ["20250816"]

aperture_radius = 15
targ_id = 2086830502105850240

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
    mask = (mag < 21)
    times = times[mask]
    mag = mag[mask]
    mag_err = mag_err[mask]
    artificial_flux = 10 ** (-0.4 * mag)
    artificial_flux_err = np.abs(10 ** (-0.4 * (mag + mag_err)) - artificial_flux)

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
            # calculate a weighted mean flux and convert back to magnitude
            mean_flux = np.sum(artificial_flux[mask] / artificial_flux_err[mask] ** 2) / np.sum(1 / artificial_flux_err[mask] ** 2)
            binned_mag.append(-2.5 * np.log10(mean_flux))
            binned_flux_err = np.sqrt(1 / np.sum(1 / artificial_flux_err[mask] ** 2))
            binned_mag_errs.append(2.5 / np.log(10) * binned_flux_err / mean_flux)
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

total_times = np.array(total_times) - 2460900  # Convert to BJD - 2460900
total_binned_times = np.array(total_binned_times) - 2460900  # Convert to BJD - 2460900
total_mags = np.array(total_mags)
total_binned_mags = np.array(total_binned_mags)
total_mag_errs = np.array(total_mag_errs)
total_binned_mag_errs = np.array(total_binned_mag_errs)

# convert to flux
artificial_flux = 10 ** (-0.4 * total_mags)
artificial_flux_err = np.abs(10 ** (-0.4 * (total_mags + total_mag_errs)) - artificial_flux)
relative_flux = artificial_flux / np.median(artificial_flux[:300])
relative_flux_err = artificial_flux_err / np.median(artificial_flux[:300])
artificial_binned_flux = 10 ** (-0.4 * total_binned_mags)
artificial_binned_flux_err = np.abs(10 ** (-0.4 * (total_binned_mags + total_binned_mag_errs)) - artificial_binned_flux)
relative_binned_flux = artificial_binned_flux / np.median(artificial_flux[:300])
relative_binned_flux_err = artificial_binned_flux_err / np.median(artificial_flux[:300])

# save relative flux to csv
output_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/kepler-670/")
output_path.mkdir_p()
output_file = output_path / f"w1m.csv"
# round to 6 decimal places
np.savetxt(output_file, np.transpose([total_times + 2460900, relative_flux, relative_flux_err]), delimiter=",", header="time,flux,flux_err", comments="", fmt="%.6f")

plt.errorbar(total_times, relative_flux, relative_flux_err, fmt="o", markersize=3, color="red", alpha=0.2, label="Unbinned Data", capsize=2)
plt.errorbar(total_binned_times, relative_binned_flux, relative_binned_flux_err, fmt="o", markersize=3, color="black", label="Binned Data", capsize=2)
plt.xlabel("Time (BJD - 2460900)")
plt.ylabel("Relative Flux")
plt.title(f"Kepler-670")
plt.grid()
plt.show()