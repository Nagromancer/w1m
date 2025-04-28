from path import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

bin_size = 5 / 1440

# Load your data
lc_dir = Path("/Users/nagro/PycharmProjects/w1m/sdss1234_lightcurves")

# tom's data
lc_files = sorted(lc_dir.glob("*tom.dat"))
tom_time = []
tom_flux = []
tom_flux_err = []
tom_binned_time = []
tom_binned_flux = []
tom_binned_flux_err = []

for file in lc_files:
    tom_time_i, mag_i, mag_err_i = np.loadtxt(file, unpack=True, usecols=(0, 2, 3), skiprows=0)
    tom_flux_i = 10 ** (-0.4 * mag_i)
    tom_flux_err_i = np.abs(tom_flux_i - 10 ** (-0.4 * (mag_i + mag_err_i)))
    tom_flux_median = np.median(tom_flux_i)
    tom_flux_i /= tom_flux_median
    tom_flux_err_i /= tom_flux_median

    # non-outlier range is between 0.5 and 1.5
    outlier_mask = (0.5 < tom_flux_i) & (tom_flux_i < 1.5)
    tom_flux_i = tom_flux_i[outlier_mask]
    tom_flux_err_i = tom_flux_err_i[outlier_mask]
    tom_time_i = tom_time_i[outlier_mask]

    tom_time.extend(tom_time_i)
    tom_flux.extend(tom_flux_i)
    tom_flux_err.extend(tom_flux_err_i)

    # bin the light curve
    tom_binned_time_i = []
    tom_binned_flux_i = []
    tom_binned_flux_err_i = []
    for i in range(int(np.ceil(tom_time_i[0] / bin_size)), int(np.ceil(tom_time_i[-1] / bin_size))):
        mask = (tom_time_i > i * bin_size) & (tom_time_i < (i + 1) * bin_size)
        if np.sum(mask) == 0:
            continue
        tom_binned_time_i.append(np.mean(tom_time_i[mask]))
        tom_binned_flux_i.append(np.mean(tom_flux_i[mask]))
        tom_binned_flux_err_i.append(np.sqrt(np.sum(tom_flux_err_i[mask] ** 2)) / np.sum(mask))

    tom_binned_time.extend(tom_binned_time_i)
    tom_binned_flux.extend(tom_binned_flux_i)
    tom_binned_flux_err.extend(tom_binned_flux_err_i)

tom_time = np.array(tom_time)
tom_flux = np.array(tom_flux)
tom_flux_err = np.array(tom_flux_err)

tom_binned_time = np.array(tom_binned_time)
tom_binned_flux = np.array(tom_binned_flux)
tom_binned_flux_err = np.array(tom_binned_flux_err)

# tnt data
lc_files = sorted(lc_dir.glob("*uspec_kg5.dat"))
tnt_time = []
tnt_flux = []
tnt_flux_err = []
tnt_binned_time = []
tnt_binned_flux = []
tnt_binned_flux_err = []

for file in lc_files:
    tnt_time_i, mag_i, mag_err_i = np.loadtxt(file, unpack=True, usecols=(0, 2, 3), skiprows=0)
    tnt_flux_i = 10 ** (-0.4 * mag_i)
    tnt_flux_err_i = np.abs(tnt_flux_i - 10 ** (-0.4 * (mag_i + mag_err_i)))
    tnt_flux_median = np.median(tnt_flux_i)
    tnt_flux_i /= tnt_flux_median
    tnt_flux_err_i /= tnt_flux_median

    # non-outlier range is between 0.5 and 1.5
    outlier_mask = (0.5 < tnt_flux_i) & (tnt_flux_i < 1.5)
    tnt_flux_i = tnt_flux_i[outlier_mask]
    tnt_flux_err_i = tnt_flux_err_i[outlier_mask]
    tnt_time_i = tnt_time_i[outlier_mask]

    tnt_time.extend(tnt_time_i)
    tnt_flux.extend(tnt_flux_i)
    tnt_flux_err.extend(tnt_flux_err_i)

    # bin the light curve
    tnt_binned_time_i = []
    tnt_binned_flux_i = []
    tnt_binned_flux_err_i = []
    for i in range(int(np.ceil(tnt_time_i[0] / bin_size)), int(np.ceil(tnt_time_i[-1] / bin_size))):
        mask = (tnt_time_i > i * bin_size) & (tnt_time_i < (i + 1) * bin_size)
        if np.sum(mask) == 0:
            continue
        tnt_binned_time_i.append(np.mean(tnt_time_i[mask]))
        tnt_binned_flux_i.append(np.mean(tnt_flux_i[mask]))
        tnt_binned_flux_err_i.append(np.sqrt(np.sum(tnt_flux_err_i[mask] ** 2)) / np.sum(mask))

    tnt_binned_time.extend(tnt_binned_time_i)
    tnt_binned_flux.extend(tnt_binned_flux_i)
    tnt_binned_flux_err.extend(tnt_binned_flux_err_i)

tnt_time = np.array(tnt_time)
tnt_flux = np.array(tnt_flux)
tnt_flux_err = np.array(tnt_flux_err)

tnt_binned_time = np.array(tnt_binned_time)
tnt_binned_flux = np.array(tnt_binned_flux)
tnt_binned_flux_err = np.array(tnt_binned_flux_err)

# W1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_5.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_binned_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_binned_lc_5.csv")
w1m_binned_lc = np.genfromtxt(w1m_binned_lc_path, delimiter=",", names=True)

w1m_time = w1m_lc["Time_HJD"]
w1m_flux = w1m_lc["Flux"]
w1m_flux_err = w1m_lc["Error"]

w1m_binned_time = w1m_binned_lc["Time_HJD"]
w1m_binned_flux = w1m_binned_lc["Flux"]
w1m_binned_flux_err = w1m_binned_lc["Error"]

# non-outlier range is between 0.5 and 1.5
outlier_mask = (0.5 < w1m_flux) & (w1m_flux < 1.5)
w1m_flux = w1m_flux[outlier_mask]
w1m_flux_err = w1m_flux_err[outlier_mask]
w1m_time = w1m_time[outlier_mask]

# Remove outliers
w1m_binned_outlier_mask = (0.5 < w1m_binned_flux) & (w1m_binned_flux < 1.5)
w1m_binned_flux = w1m_binned_flux[w1m_binned_outlier_mask]
w1m_binned_flux_err = w1m_binned_flux_err[w1m_binned_outlier_mask]
w1m_binned_time = w1m_binned_time[w1m_binned_outlier_mask]

# combine the two datasets
time = np.concatenate((tom_time, w1m_time, tnt_time))
flux = np.concatenate((tom_flux, w1m_flux, tnt_flux))
flux_err = np.concatenate((tom_flux_err, w1m_flux_err, tnt_flux_err))

binned_time = np.concatenate((tom_binned_time, w1m_binned_time, tnt_binned_time))
binned_flux = np.concatenate((tom_binned_flux, w1m_binned_flux, tnt_binned_flux))
binned_flux_err = np.concatenate((tom_binned_flux_err, w1m_binned_flux_err, tnt_binned_flux_err))

min_time = 2460710.292987
time_delta = 4.92 / 24

binned_time -= min_time - time_delta  # convert to days since first observation
time -= min_time - time_delta  # convert to days since first observation
end_time = np.max(time)

period = 14.804 / 24

offset = 2 * period / 10
folded_time = time / period
folded_binned_time = binned_time / period

tom_time -= min_time - time_delta  # convert to days since first observation
tnt_time -= min_time - time_delta  # convert to days since first observation
w1m_time -= min_time - time_delta  # convert to days since first observation

tom_binned_time -= min_time - time_delta
tnt_binned_time -= min_time - time_delta  # convert to days since first observation
w1m_binned_time -= min_time - time_delta  # convert to days since first observation

folded_tom_time = tom_time / period
folded_tom_binned_time = tom_binned_time / period
folded_tnt_time = tnt_time / period
folded_tnt_binned_time = tnt_binned_time / period
folded_w1m_time = w1m_time / period
folded_w1m_binned_time = w1m_binned_time / period

fig, ax1 = plt.subplots(figsize=(15, 20))

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
colours = [blue, orange, pink]

plt.errorbar((folded_w1m_time % 1) - 0.5, w1m_flux - 1 + offset * np.floor(folded_w1m_time),
             yerr=w1m_flux_err, fmt='o', color=colours[0], markersize=5, alpha=0.1)
plt.errorbar((folded_w1m_binned_time % 1) - 0.5, w1m_binned_flux - 1 + offset * np.floor(folded_w1m_binned_time),
             yerr=w1m_binned_flux_err, fmt='o', color=colours[0], markersize=5, label='W1m')
plt.errorbar((folded_tnt_time % 1) - 0.5, tnt_flux - 1 + offset * np.floor(folded_tnt_time),
             yerr=tnt_flux_err, fmt='^', color=colours[1], markersize=5, alpha=0.02)
plt.errorbar((folded_tnt_binned_time % 1) - 0.5, tnt_binned_flux - 1 + offset * np.floor(folded_tnt_binned_time),
             yerr=tnt_binned_flux_err, fmt='^', color=colours[1], markersize=5, label='TNT')
plt.errorbar((folded_tom_time % 1) - 0.5, tom_flux - 1 + offset * np.floor(folded_tom_time),
             yerr=tom_flux_err, fmt='s', color=colours[2], markersize=5, alpha=0.1)
plt.errorbar((folded_tom_binned_time % 1) - 0.5, tom_binned_flux - 1 + offset * np.floor(folded_tom_binned_time),
             yerr=tom_binned_flux_err, fmt='s', color=colours[2], markersize=5, label='Tom')

# subtract one from the ax1 y ticks and y tick labels

ax1.set_title(f"SDSS1234+5606", pad=30)
ax1.set_xlabel(f"Phase")
ax1.set_ylabel(f"Relative Flux $+$ Offset")
ax1.legend(loc=(0.75, 0.69), fontsize=32)
ax1.set_xlim(-0.55, 0.55)
lower_lim = -0.5
upper_lim = offset * np.floor(end_time / period) + 0.5
ax1.set_yticks(np.arange(0, upper_lim, 2))
ax1.grid()
ax1.set_ylim(lower_lim, upper_lim)
ax2 = ax1.twinx()
ax2.set_ylabel("$T-T_0$ (days)", labelpad=10)
ax2.set_ylim(lower_lim, upper_lim)

# show phase in hours at the top
ax3 = ax1.twiny()
ax3.set_xlim(-0.55, 0.55)
ax3.set_xlabel(f"Phase (hours)", labelpad=10)
x_ticks_phase_hours = np.arange(int(np.ceil(-period * 12))-1, period * 12 + 2, 2)
x_ticks_phase = x_ticks_phase_hours / (period * 24)
ax3.set_xticks(x_ticks_phase)
ax3.set_xticklabels([int(x_tick) for x_tick in x_ticks_phase_hours])

y_ticks_time = np.arange(0, end_time, 10)
y_ticks_flux = (y_ticks_time * offset / period)
ax2.set_yticks(y_ticks_flux)
ax2.set_yticklabels([int(y_tick) for y_tick in y_ticks_time])
ax1.axvline(0., color='r', linestyle='--')
plt.subplots_adjust(left=0.08, right=0.91, bottom=0.06, top=0.9)
plt.savefig("combined_phase_plot.pdf", dpi=100)
plt.show()
