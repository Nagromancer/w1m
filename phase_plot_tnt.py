from path import Path
import numpy as np
import matplotlib.pyplot as plt
from utilities import bin_data

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

bin_size = 5 / 1440

# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_w1m.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["Time_BJD"]
w1m_flux = w1m_lc["Flux"]
w1m_flux_err = w1m_lc["Error"]

# tom's data
tom_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_tom.csv")
tom_lc = np.genfromtxt(tom_lc_path, delimiter=",", names=True)
tom_time = tom_lc["Time_BJD"]
tom_flux = tom_lc["Flux"]
tom_flux_err = tom_lc["Error"]

# tnt data
tnt_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_tnt.csv")
tnt_lc = np.genfromtxt(tnt_lc_path, delimiter=",", names=True)
tnt_time = tnt_lc["Time_BJD"]
tnt_flux = tnt_lc["Flux"]
tnt_flux_err = tnt_lc["Error"]


# combine the two datasets
time = np.concatenate((tom_time, w1m_time, tnt_time))
flux = np.concatenate((tom_flux, w1m_flux, tnt_flux))
flux_err = np.concatenate((tom_flux_err, w1m_flux_err, tnt_flux_err))

min_time = 2460710.092

time -= min_time  # convert to days since first observation
end_time = np.max(time)

period = 14.803 / 24

offset = 2 * period / 10
folded_time = time / period

tom_time -= min_time  # convert to days since first observation
tnt_time -= min_time  # convert to days since first observation
w1m_time -= min_time  # convert to days since first observation

tom_binned_time, tom_binned_flux, tom_binned_flux_errs = bin_data(bin_size, tom_time, tom_flux, tom_flux_err)
tnt_binned_time, tnt_binned_flux, tnt_binned_flux_errs = bin_data(bin_size, tnt_time, tnt_flux, tnt_flux_err)
w1m_binned_time, w1m_binned_flux, w1m_binned_flux_errs = bin_data(bin_size, w1m_time, w1m_flux, w1m_flux_err)

folded_tom_time = tom_time / period
tom_binned_time = tom_binned_time / period
folded_tnt_time = tnt_time / period
tnt_binned_time = tnt_binned_time / period
folded_w1m_time = w1m_time / period
w1m_binned_time = w1m_binned_time / period

fig, ax1 = plt.subplots(figsize=(15, 20))

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
colours = [blue, orange, pink]

plt.errorbar((folded_w1m_time % 1) - 0.5, w1m_flux - 1 + offset * np.floor(folded_w1m_time),
             yerr=w1m_flux_err, fmt='o', color=colours[0], markersize=5, alpha=0.1)
plt.errorbar((w1m_binned_time % 1) - 0.5, w1m_binned_flux - 1 + offset * np.floor(w1m_binned_time),
             yerr=w1m_binned_flux_errs, fmt='o', color=colours[0], markersize=5, label='W1m')
plt.errorbar((folded_tnt_time % 1) - 0.5, tnt_flux - 1 + offset * np.floor(folded_tnt_time),
             yerr=tnt_flux_err, fmt='^', color=colours[1], markersize=5, alpha=0.02)
plt.errorbar((tnt_binned_time % 1) - 0.5, tnt_binned_flux - 1 + offset * np.floor(tnt_binned_time),
             yerr=tnt_binned_flux_errs, fmt='^', color=colours[1], markersize=5, label='TNT')
plt.errorbar((folded_tom_time % 1) - 0.5, tom_flux - 1 + offset * np.floor(folded_tom_time),
             yerr=tom_flux_err, fmt='s', color=colours[2], markersize=5, alpha=0.1)
plt.errorbar((tom_binned_time % 1) - 0.5, tom_binned_flux - 1 + offset * np.floor(tom_binned_time),
             yerr=tom_binned_flux_errs, fmt='s', color=colours[2], markersize=5, label='Tom')

# subtract one from the ax1 y ticks and y tick labels

ax1.set_title(f"SDSS1234+5606", pad=30)
ax1.set_xlabel(f"Phase")
ax1.set_ylabel(f"Relative Flux $+$ Offset")
ax1.legend(loc=(0.75, 0.63), fontsize=32)
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
