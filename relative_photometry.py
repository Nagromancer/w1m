from path import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import wotan
from astropy.coordinates import SkyCoord


def get_light_curve(phot, target_id, aperture_radius):
    target_phot = phot[phot["ID"] == target_id]
    time = np.array(target_phot["HJD"])
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
bin_size = 5 / 1440  # 10 minutes in days
dates = ["20250205", "20250206", "20250207", "20250208", "20250213", "20250216", "20250218", "20250219", "20250220", "20250221", "20250222", "20250223", "20250224", "20250225", "20250226", "20250310", "20250317", "20250318", "20250320", "20250321", "20250322", "20250324"]
aperture_radius = 5
targ_id = 1571584539980588544

total_images = 0
total_binned_times = []
total_binned_fluxes = []
total_binned_flux_errs = []
total_times = []
total_fluxes = []
total_flux_errs = []
for date in dates:
    try:
        target = f"Gaia DR3 {targ_id}"
        cat_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/reference_catalogues/{target}/{date}-{target}.fits")
        phot_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/{date}-{target}-{camera}-phot.fits")
        image_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/calibrated")
        phot_table = Table.read(phot_path)
    except FileNotFoundError:
        target = target.replace(" ", "_")
        cat_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/reference_catalogues/{target}/{date}-{target}.fits")
        phot_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/{date}-{target}-{camera}-phot.fits")
        image_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/calibrated")
        phot_table = Table.read(phot_path)

    num_images = len(image_path.files("*.fits"))
    total_images += num_images

    cat = Table.read(cat_path)
    cat = cat[np.isin(np.array(cat["ID"]), np.array(phot_table["ID"]))]

    # create a reference light curve
    std_devs = []
    binned_std_devs = []
    average_fluxes = []
    target_id = targ_id
    target_coords = SkyCoord(cat[cat["ID"] == target_id]["RA"], cat[cat["ID"] == target_id]["DEC"], unit="deg")
    target_mag = cat[cat["ID"] == target_id]['BP_MAG' if camera == 'blue' else 'RP_MAG'][0]

    ref_cat = cat[cat["VALID"]]
    ref_cat = ref_cat[ref_cat["ID"] != target_id]

    ref_cat = ref_cat[:10]
    ref_cat.pprint_all()

    times, fluxes, flux_errs = get_light_curve(phot_table, target_id, aperture_radius)
    good_phot_mask = np.ones(len(times), dtype=bool)
    times = times[good_phot_mask]
    fluxes = fluxes[good_phot_mask]
    flux_errs = flux_errs[good_phot_mask]

    ref_flux = np.zeros(num_images)
    for i, star in enumerate(ref_cat):
        _, flux, _ = get_light_curve(phot_table, star["ID"], aperture_radius)
        ref_flux += flux
    ref_flux  = ref_flux[good_phot_mask]

    times -= 2460000
    relative_flux = fluxes / ref_flux
    median_rel_flux = np.median(relative_flux)
    relative_flux /= median_rel_flux
    relative_flux_errs = flux_errs / ref_flux
    relative_flux_errs /= np.abs(median_rel_flux)

    # remove outliers
    # standard_deviation = np.std(relative_flux)
    # mask = (relative_flux - 1) < 5 * standard_deviation
    # times = times[mask]
    # relative_flux = relative_flux[mask]
    # relative_flux_errs = relative_flux_errs[mask]

    flattened, trend = wotan.flatten(time=times, flux=relative_flux, method='biweight', window_length=0.02, return_trend=True)

    # bin the light curve
    if bin_size is not None:
        binned_times = []
        binned_relative_flux = []
        binned_relative_flux_errs = []
        binned_trend = []
        for i in range(int(np.ceil(times[0] / bin_size)), int(np.ceil(times[-1] / bin_size))):
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
    average_fluxes.append(np.mean(fluxes))

    print(f"{target_mag:.2f} mag: Standard deviation: Trend - {trend_std_dev:.3f} ppt | Relative - {std_dev:.3f} ppt | Binned - {binned_std_dev:.3f} ppt")
    total_binned_times += list(binned_times)
    total_binned_fluxes += list(binned_relative_flux)
    total_binned_flux_errs += list(binned_relative_flux_errs)
    total_times += list(times)
    total_fluxes += list(relative_flux)
    total_flux_errs += list(relative_flux_errs)

print(f"Total images: {total_images}")
print(f"Total exposure: {total_images / 120:.2f} hours")
plt.errorbar(total_times, total_fluxes, yerr=total_flux_errs, fmt='o', color='black', markersize=5, alpha=0.1)
plt.errorbar(total_binned_times, total_binned_fluxes, yerr=total_binned_flux_errs, fmt='o', color='black', markersize=5)
# plt.plot(times, trend, color='red', lw=2)
plt.title(f"Target {target_id} ({target_mag:.2f} mag) - {aperture_radius} px")
plt.xlabel(f"Time (HJD - 2460000)")
plt.ylabel(f"Relative Flux ({'Blue' if camera == 'blue' else 'Red'} Camera)")
plt.show()
plt.close()

# perform BLS periodogram
from astropy.timeseries import BoxLeastSquares
model = BoxLeastSquares(total_times, total_fluxes, total_flux_errs)
# test for periods between 0.2 and 4 days
frequencies = np.linspace(0.6, 0.63, 10000)
periodogram = model.power(frequencies, 0.01, method="fast", objective="snr")
plt.plot(periodogram.period, periodogram.power)
plt.xlabel("Period (days)")
plt.ylabel("Power")
plt.title("BLS Periodogram")
plt.show()

peak_period = periodogram.period[np.argmax(periodogram.power)]
print(f"Peak period: {peak_period * 24:.6f} hours")
# peak_period = 14.842 / 24

# fold to the peak period
total_times = np.array(total_times)
total_fluxes = np.array(total_fluxes)
total_flux_errs = np.array(total_flux_errs)
total_binned_times = np.array(total_binned_times)
total_binned_fluxes = np.array(total_binned_fluxes)
total_binned_flux_errs = np.array(total_binned_flux_errs)
total_binned_times -= np.min(total_times)
total_times -= np.min(total_times)

offset = 0.1
folded_time = total_times / peak_period
folded_binned_time = total_binned_times / peak_period
fig = plt.figure(figsize=(15, 20))
plt.errorbar((folded_time % 1) * peak_period * 24, total_fluxes + offset * np.floor(folded_time), yerr=total_flux_errs, fmt='o', color='black', markersize=5, alpha=0.1)
plt.errorbar((folded_binned_time % 1) * peak_period * 24, total_binned_fluxes + offset * np.floor(folded_binned_time), yerr=total_binned_flux_errs, fmt='o', color='black', markersize=5)

plt.title(f"SDSS1234+5606")
plt.xlabel(f"Phase (hours)")
plt.ylabel(f"Relative Flux $+$ Offset")
lower_lim = 0.75
upper_lim = 9
plt.ylim(lower_lim, upper_lim)
# plt.ylim(lower_lim, 1.2)
# make a second y axis showing time, because the offset is showing time
ax2 = plt.gca().twinx()
ax2.set_ylabel("$T-T_0$ (days)")
# time is the period * phase
ax2.set_ylim(lower_lim, upper_lim)
# ax2.set_ylim(lower_lim, 1.2)

y_ticks_time = np.arange(0, 50, 5)
y_ticks_flux = (y_ticks_time * offset / peak_period) + 1
ax2.set_yticks(y_ticks_flux)
ax2.set_yticklabels(y_ticks_time)
plt.tight_layout()
plt.show()





exit()

plt.plot(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], std_devs, 'o', color='black')
plt.plot(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], binned_std_devs, 'o', color='red')
if camera == 'blue':
    plt.xlabel('${G}_\mathrm{BP}$ (mag)')
else:
    plt.xlabel('${G}_\mathrm{RP}$ (mag)')
plt.ylabel("Standard Deviation (ppt)")
plt.yscale('log')
plt.title(f"Standard Deviation vs Magnitude - {aperture_radius} px")
# plt.ylim(0.2, 200)
# plt.xlim(12.5, 18.5)
plt.grid()
plt.show()

# plot flux vs magnitude
plt.scatter(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], average_fluxes, 40, c=cat[cat["VALID"]]["BP_RP"], cmap='coolwarm', clim=(0.0, 2.5))
# colorbar with label
# fit a line to log(flux) vs magnitude with np.polyfit
trend = np.polyfit(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"], np.log10(average_fluxes), 1)
x = np.linspace(np.min(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"]), np.max(cat[cat["VALID"]]["BP_MAG" if camera == "blue" else "RP_MAG"]), 100)
y = 10 ** (trend[0] * x + trend[1])
plt.plot(x, y, color='black', lw=1)
print(f"ZP: {-trend[1] / trend[0]:.3f} mag")


plt.colorbar(label="${G}_\mathrm{BP} - G_\mathrm{RP}$ (mag)")
if camera == 'blue':
    plt.xlabel('${G}_\mathrm{BP}$ (mag)')
else:
    plt.xlabel('${G}_\mathrm{RP}$ (mag)')
plt.ylabel("Mean Flux (e$^{-}$ s$^{-1}$)")
plt.title(f"Mean Flux vs Magnitude - {aperture_radius} px")
plt.grid()
plt.yscale('log')
plt.tight_layout()
plt.show()
