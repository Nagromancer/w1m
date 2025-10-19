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

dates = ["20250112", "20250114", "20250115", "20250116", "20250117", "20250119", "20250124", "20250125", "20250126"]
dates += ["20250819", "20250820", "20250821", "20250822", "20250826"]
dates += ["20250827", "20250828", "20250829"]
dates += ["20250831", "20250901", "20250902", "20250903", "20250904", "20250905"]
dates += ["20250907", "20250908", "20250909", "20250910", "20250911", "20250912", "20250915", "20250916", "20250917", "20250921", "20250925",
          "20250926", "20250927", "20250929", "20250930", "20251001", "20251002", "20251006", "20251008", "20251009", "20251010",
          "20251011", "20251012"]

print(f"Number of dates: {len(dates)}")

aperture_radius = 10
bin_size = 10 / 1440  # 10 minutes in days
targ_id = 380316777081879680

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

total_times = np.array(total_times)
total_binned_times = np.array(total_binned_times)
total_mags = np.array(total_mags)
total_binned_mags = np.array(total_binned_mags)
total_mag_errs = np.array(total_mag_errs)
total_binned_mag_errs = np.array(total_binned_mag_errs)


early_data = total_times < 2460900
early_times = total_times[early_data]
early_mags = total_mags[early_data]
early_mag_errs = total_mag_errs[early_data]
early_data = total_binned_times < 2460900
early_binned_times = total_binned_times[early_data]
early_binned_mags = total_binned_mags[early_data]
early_binned_mag_errs = total_binned_mag_errs[early_data]

late_data = total_times >= 2460900
late_times = total_times[late_data]
late_mags = total_mags[late_data]
late_mag_errs = total_mag_errs[late_data]
late_binned_data = total_binned_times >= 2460900
late_binned_times = total_binned_times[late_binned_data]
late_binned_mags = total_binned_mags[late_binned_data]
late_binned_mag_errs = total_binned_mag_errs[late_binned_data]

early_times -= 2460000
late_times -= 2460000
early_binned_times -= 2460000
late_binned_times -= 2460000


# total_binned_times = total_binned_times[late_binned_data]
# total_binned_mags = total_binned_mags[late_binned_data]
# total_binned_mag_errs = total_binned_mag_errs[late_binned_data]
# total_times = total_times[late_data]
# total_mags = total_mags[late_data]
# total_mag_errs = total_mag_errs[late_data]


# plt.errorbar(early_binned_times, early_binned_mags, yerr=early_binned_mag_errs,
#                 fmt='.', color='k')
plt.errorbar(late_binned_times, late_binned_mags, yerr=late_binned_mag_errs,
                fmt='.', color='k')
plt.xlabel("Time (BJD - 2460000)")
plt.ylabel(r"$G_\mathrm{BP}$ (mag)")
plt.gca().invert_yaxis()
plt.title("HS0019+3947")
plt.grid()
plt.show()


ratio = (late_binned_times.max() - late_binned_times.min() + 1) / (early_binned_times.max() - early_binned_times.min() + 1)
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(14, 9),
    gridspec_kw={'width_ratios': [1, ratio]}
)

ax2.errorbar(late_binned_times, late_binned_mags, yerr=late_binned_mag_errs,
             fmt='.', color='k')
ax2.set_xlim(late_times.min()-0.5, late_times.max()+0.5)

# --- Shared y-label ---
ax1.set_ylabel(r"$G_\mathrm{BP}$ (mag)")


# --- Hide the joined spines (to indicate axis break) ---
# Hide spines
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
d = .01  # size of break markers

# Hide ticks on the broken edges
ax2.tick_params(labelleft=False, left=False)    # no ticks on ax2’s left

# Add diagonal slashes at the break
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d*ratio, 1+d*ratio), (-d, +d), **kwargs)        # lower diagonal
ax1.plot((1-d*ratio, 1+d*ratio), (1-d, 1+d), **kwargs)      # upper diagonal


kwargs.update(transform=ax2.transAxes)  # switch to right subplot
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax1.set_xlim(early_times.min()-0.5, early_times.max()+0.5)
ax1.errorbar(early_binned_times, early_binned_mags, yerr=early_binned_mag_errs,
             fmt='.', color='k', label="Binned")
ax1.set_xticks([690, 700])
ax1.tick_params(labelright=False, right=False)  # no ticks on ax1’s right
# invert y axis
ax1.invert_yaxis()
ax1.grid()

# x axis label
fig.text(0.5, 0.04, "Time (BJD - 2460000)", ha='center', va='center', fontsize=32)

# title
fig.suptitle("HS0019+3947", fontsize=32)
fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.13, wspace=0.05)

ax2.grid()

plt.show()



# do a ls periodogram on total_binned_times and total_binned_mags


from astropy.timeseries import LombScargle
frequency = np.linspace(0.0005, 0.05, 100000)
power = LombScargle(total_times, total_mags).power(frequency)
plt.plot(1/frequency, power)
plt.xscale('log')
plt.xlabel('Period (days)')
plt.ylabel('Lomb-Scargle Power')
plt.title('Lomb-Scargle Periodogram of HS0019+3947')
plt.grid()
plt.show()

peak_freq = frequency[np.argmax(power)]
print(f"Peak period: {1/peak_freq} days")

# phase fold the light curve on the peak period
period = 1/peak_freq
phase = (total_binned_times % period) / period

# shift so 0 phase is at min mag
min_phase = phase[np.argmin(total_binned_mags)]
phase = (phase - min_phase) % 1

phase = np.concatenate([phase-1, phase, phase+1])
plot_binned_mags = np.concatenate([total_binned_mags, total_binned_mags, total_binned_mags])
plot_binned_mag_errs = np.concatenate([total_binned_mag_errs, total_binned_mag_errs, total_binned_mag_errs])

plt.errorbar(phase, plot_binned_mags, yerr=plot_binned_mag_errs, fmt='.', color='k')
# light grey fill between -0.5 and 0 and between 1 and 1.5
plt.fill_betweenx([13, 21], -0.5, 0, color='lightgrey', alpha=0.5)
plt.fill_betweenx([13, 21], 1, 1.5, color='lightgrey', alpha=0.5)
plt.ylim(14.2, 19.8)
plt.xlim(-0.5, 1.5)
plt.xlabel('Phase')
plt.ylabel(r"$G_\mathrm{BP}$ (mag)")
plt.title(f'Phase Folded Light Curve of HS0019+3947 (P={period:.4f} days)')
plt.gca().invert_yaxis()
plt.grid()
plt.show()