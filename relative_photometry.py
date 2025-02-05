from path import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import wotan
from astropy.coordinates import SkyCoord


def get_light_curve(phot, target_id, aperture_radius):
    target_phot = phot[phot["ID"] == target_id]
    time = np.array(target_phot["BJD"])
    flux = np.array(target_phot[f"FLUX_{aperture_radius}"])
    flux_err = np.array(target_phot[f"FLUX_ERR_{aperture_radius}"])
    return time, flux, flux_err


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32

camera = "blue"
bin_size = 10 / 1440  # 10 minutes in days
date = "20250128"

aperture_radius = 20

cat_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/reference_catalogues/Gaia DR3 817461778282929664/{date}-Gaia DR3 817461778282929664.fits")
phot_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/Gaia DR3 817461778282929664/{date}-Gaia DR3 817461778282929664-{camera}-phot.fits")
image_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/Gaia DR3 817461778282929664/calibrated")
phot_table = Table.read(phot_path)

num_images = len(image_path.files("*.fits"))

cat = Table.read(cat_path)

# correct catalogue by including only stars with as many measurements as there are images
cat = cat[np.isin(np.array(cat["ID"]), np.array(phot_table["ID"]))]
cat[cat["VALID"]].pprint_all()

# create a reference light curve from the sum of all stars in the reference catalogue
std_devs = []
binned_std_devs = []
for target_id in cat[cat["VALID"]]["ID"]:
    target_coords = SkyCoord(cat[cat["ID"] == target_id]["RA"], cat[cat["ID"] == target_id]["DEC"], unit="deg")
    target_mag = cat[cat["ID"] == target_id]['BP_MAG' if camera == 'blue' else 'RP_MAG'][0]

    ref_cat = cat[cat["VALID"]]
    ref_cat = ref_cat[ref_cat["ID"] != target_id]

    # ref_coords = SkyCoord(ref_cat["RA"], ref_cat["DEC"], unit="deg")
    # ref_cat["SEPARATION"] = target_coords.separation(ref_coords).to("arcmin")
    # ref_cat = ref_cat[ref_cat['BP_MAG' if camera == 'blue' else 'RP_MAG'] < target_mag + 2.0]
    # ref_cat.sort("SEPARATION")

    ref_cat = ref_cat[:10]

    ref_flux = np.zeros(num_images)
    for i, star in enumerate(ref_cat):
        _, flux, _ = get_light_curve(phot_table, star["ID"], aperture_radius)
        ref_flux += flux

    times, fluxes, flux_errs = get_light_curve(phot_table, target_id, aperture_radius)
    times -= times[0]
    relative_flux = fluxes / ref_flux
    median_rel_flux = np.median(relative_flux)
    relative_flux /= median_rel_flux
    relative_flux_errs = flux_errs / ref_flux
    relative_flux_errs /= median_rel_flux

    # remove outliers
    standard_deviation = np.std(relative_flux)
    mask = np.abs(relative_flux - 1) < 3 * standard_deviation
    times = times[mask]
    relative_flux = relative_flux[mask]
    relative_flux_errs = relative_flux_errs[mask]

    flattened, trend = wotan.flatten(time=times, flux=relative_flux, method='biweight', window_length=0.1, return_trend=True)

    # bin the light curve
    if bin_size is not None:
        binned_times = []
        binned_relative_flux = []
        binned_relative_flux_errs = []
        binned_trend = []
        for i in range(int(np.ceil(times[-1] / bin_size))):
            mask = (times > i * bin_size) & (times < (i + 1) * bin_size)
            if np.sum(mask) == 0:
                continue
            binned_times.append(np.mean(times[mask]))
            binned_relative_flux.append(np.mean(relative_flux[mask]))
            binned_relative_flux_errs.append(np.sqrt(np.sum(relative_flux_errs[mask] ** 2)) / np.sum(mask))
            binned_trend.append(np.mean(trend[mask]))
    else:
        binned_times = times
        binned_relative_flux = relative_flux
        binned_relative_flux_errs = relative_flux_errs
        binned_trend = trend
    binned_times = np.array(binned_times)
    binned_relative_flux = np.array(binned_relative_flux)
    binned_relative_flux_errs = np.array(binned_relative_flux_errs)
    binned_trend = np.array(binned_trend)

    std_dev = np.std(relative_flux / trend) * 1000
    trend_std_dev = np.std(trend) * 1000
    std_devs.append(std_dev)
    binned_std_dev = np.std(binned_relative_flux / binned_trend) * 1000
    binned_std_devs.append(binned_std_dev)

    print(f"{target_mag:.2f} mag: Standard deviation: Trend - {trend_std_dev:.3f} ppt | Relative - {std_dev:.3f} ppt | Binned - {binned_std_dev:.3f} ppt")
    plt.errorbar(times, relative_flux, yerr=relative_flux_errs, fmt='o', color='black', markersize=5, alpha=0.2)
    plt.errorbar(binned_times, binned_relative_flux, yerr=binned_relative_flux_errs, fmt='o', color='black', markersize=5)
    plt.plot(times, trend, color='red', lw=2)
    plt.title(f"Target {target_id} ({target_mag:.2f} mag) - {aperture_radius} px")
    plt.xlabel("Time (days)")
    plt.ylabel(f"Relative Flux ({'BP' if camera == 'blue' else 'RP'})")
    plt.show()
    plt.close()

plt.plot(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], std_devs, 'o', color='black')
plt.plot(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], binned_std_devs, 'o', color='red')
plt.xlabel(f"{'BP' if camera == 'blue' else 'RP'} Magnitude")
plt.ylabel("Standard Deviation (ppt)")
plt.yscale('log')
plt.title(f"Standard Deviation vs Magnitude - {aperture_radius} px")
plt.ylim(0.2, 200)
plt.xlim(12.5, 18.5)
plt.grid()
plt.show()