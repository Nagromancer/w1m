from path import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32

# Load your data
lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_5.csv")
lc = np.genfromtxt(lc_path, delimiter=",", names=True)
binned_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_binned_lc_5.csv")
binned_lc = np.genfromtxt(binned_lc_path, delimiter=",", names=True)

time = lc["Time_HJD"]
flux = lc["Flux"]
flux_err = lc["Error"]

binned_time = binned_lc["Time_HJD"]
binned_flux = binned_lc["Flux"]
binned_flux_err = binned_lc["Error"]

# non-outlier range is between 0.5 and 1.5
outlier_mask = (0.5 < flux) & (flux < 1.5)
flux = flux[outlier_mask]
flux_err = flux_err[outlier_mask]
time = time[outlier_mask]

# Remove outliers
binned_outlier_mask = (0.5 < binned_flux) & (binned_flux < 1.5)
binned_flux = binned_flux[binned_outlier_mask]
binned_flux_err = binned_flux_err[binned_outlier_mask]
binned_time = binned_time[binned_outlier_mask]

time_delta = - 1.03 / 24

binned_time -= time[0] - time_delta  # convert to days since first observation
time -= time[0] - time_delta  # convert to days since first observation

period = 14.8034 / 24

# Plot setup
offset = 0.075
folded_time = time / period
folded_phase = (folded_time % 1) * period * 24

folded_binned_time = binned_time / period
folded_binned_phase = (folded_binned_time % 1) * period * 24

# === Step 1: Find the big gap automatically ===

gap_threshold = 20  # days, adjust if needed
time_diffs = np.diff(time)
big_gaps = np.where(time_diffs > gap_threshold)[0]

if len(big_gaps) == 0:
    raise ValueError("No large gaps detected!")

gap_idx = big_gaps[0]

buffer = 0.5
bottom_low = 1 - buffer
bottom_high = offset * np.floor(time[gap_idx] / period) + 1 + buffer
bottom_high_time = (bottom_high - 1) / offset * period
bottom_max_tick_time = int(np.floor(bottom_high_time / 5) * 5 + 5)

top_low = offset * np.ceil(time[gap_idx + 1] / period) + 1 - buffer
top_low_time = (top_low - 1) / offset * period
top_min_tick_time = int(np.floor(top_low_time / 5) * 5 + 5)
top_high = offset * np.floor(time[-1] / period) + 1 + buffer
top_high_time = (top_high - 1) / offset * period
top_max_tick_time = int(np.floor(top_high_time / 5) * 5 + 5)

height_ratio = (top_high - top_low) / (bottom_high - bottom_low)
print(f"Bottom plot from {bottom_low} to {bottom_high}")
print(f"Top plot from {top_low} to {top_high}")

# We'll split data into two parts
before_gap = np.arange(len(time)) <= gap_idx
after_gap = np.arange(len(time)) > gap_idx

first_time_end = time[before_gap][-1]
print(f"First time end: {first_time_end}")

first_region_time_range = (time[before_gap][0], time[before_gap][-1])
second_region_time_range = (time[after_gap][0], time[after_gap][-1])

# plot
offset = 0.075
folded_time = time / period
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 20),
                               gridspec_kw={'height_ratios': [height_ratio, 1], 'hspace': 0.02})

fig.subplots_adjust(0.06, 0.05, 0.92, 0.95)  # adjust space between Axes

ax1.errorbar((folded_time % 1) * period * 24, flux + offset * np.floor(folded_time), yerr=flux_err, fmt='o', color='black', markersize=5, alpha=0.05)
ax1.errorbar((folded_binned_time % 1) * period * 24, binned_flux + offset * np.floor(folded_binned_time), yerr=binned_flux_err, fmt='o', color='black', markersize=5, alpha=0.5)
ax2.errorbar((folded_time % 1) * period * 24, flux + offset * np.floor(folded_time), yerr=flux_err, fmt='o', color='black', markersize=5, alpha=0.05)
ax2.errorbar((folded_binned_time % 1) * period * 24, binned_flux + offset * np.floor(folded_binned_time), yerr=binned_flux_err, fmt='o', color='black', markersize=5, alpha=0.5)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.spines.bottom.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.spines.top.set_visible(False)
ax2.tick_params(labeltop=False)  # don't put tick labels at the top

ax2.set_ylim(bottom_low, bottom_high)
ax1.set_ylim(top_low, top_high)

# keep only integer ax2 yticks
y_ticks = np.arange(int(np.ceil(bottom_low)), int(np.ceil(bottom_high)), 1)
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_ticks)
y_ticks = np.arange(int(np.ceil(top_low)), int(np.ceil(top_high)), 1)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_ticks)

fig.suptitle(f"SDSS1234+5606", fontsize=32)
fig.supxlabel(f"Phase (hours)", fontsize=32, y=0.0)
# fig.supylabel("$T-T_0$ (days)", fontsize=32, x=0.97)
fig.text(0.97, 0.5, "$T-T_0$ (days)", va='center', rotation=90, fontsize=32)
fig.supylabel(f"Relative Flux $+$ Offset", fontsize=32, x=0.01)

# plt.show()

ax21 = ax2.twinx()
ax21.set_ylim(bottom_low, bottom_high)

ax21.spines.top.set_visible(False)
ax21.tick_params(labeltop=False)  # don't put tick labels at the top

y21_ticks_time = np.arange(0, bottom_max_tick_time, 5)
y21_ticks_flux = (y21_ticks_time * offset / period) + 1
ax21.set_yticks(y21_ticks_flux)
ax21.set_yticklabels(y21_ticks_time)

ax11 = ax1.twinx()
ax11.set_ylim(top_low, top_high)

ax11.spines.bottom.set_visible(False)
ax11.xaxis.tick_top()
ax11.tick_params(labeltop=False)

y11_ticks_time = np.arange(top_min_tick_time, top_max_tick_time, 5)
y11_ticks_flux = (y11_ticks_time * offset / period) + 1
ax11.set_yticks(y11_ticks_flux)
ax11.set_yticklabels(y11_ticks_time)
ax1.axvline(x=7.4, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=7.4, color='red', linestyle='--', linewidth=2)

plt.show()
