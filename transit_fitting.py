from path import Path
import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

period = 14.804 / 24
ephemeris = 2460710.292987 - 4.92 / 24 + period / 2

w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_5.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_binned_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_binned_lc_5.csv")
w1m_binned_lc = np.genfromtxt(w1m_binned_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["Time_HJD"] - ephemeris
w1m_flux = w1m_lc["Flux"]
w1m_flux_err = w1m_lc["Error"]
w1m_outlier_mask = (0.5 < w1m_flux) & (w1m_flux < 1.5)
w1m_flux = w1m_flux[w1m_outlier_mask]
w1m_flux_err = w1m_flux_err[w1m_outlier_mask]
w1m_time = w1m_time[w1m_outlier_mask]
w1m_binned_time = w1m_binned_lc["Time_HJD"] - ephemeris
w1m_binned_flux = w1m_binned_lc["Flux"]
w1m_binned_flux_err = w1m_binned_lc["Error"]
w1m_binned_outlier_mask = (0.5 < w1m_binned_flux) & (w1m_binned_flux < 1.5)
w1m_binned_flux = w1m_binned_flux[w1m_binned_outlier_mask]
w1m_binned_flux_err = w1m_binned_flux_err[w1m_binned_outlier_mask]
w1m_binned_time = w1m_binned_time[w1m_binned_outlier_mask]

tom_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_tom.csv")
tom_lc = np.genfromtxt(tom_lc_path, delimiter=",", names=True)
tom_binned_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_binned_lc_tom.csv")
tom_binned_lc = np.genfromtxt(tom_binned_lc_path, delimiter=",", names=True)
tom_time = tom_lc["Time_HJD"] - ephemeris
tom_flux = tom_lc["Flux"]
tom_flux_err = tom_lc["Error"]
tom_binned_time = tom_binned_lc["Time_HJD"] - ephemeris
tom_binned_flux = tom_binned_lc["Flux"]
tom_binned_flux_err = tom_binned_lc["Error"]

tnt_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_tnt.csv")
tnt_lc = np.genfromtxt(tnt_lc_path, delimiter=",", names=True)
tnt_binned_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_binned_lc_tnt.csv")
tnt_binned_lc = np.genfromtxt(tnt_binned_lc_path, delimiter=",", names=True)
tnt_time = tnt_lc["Time_HJD"] - ephemeris
tnt_flux = tnt_lc["Flux"]
tnt_flux_err = tnt_lc["Error"]
tnt_binned_time = tnt_binned_lc["Time_HJD"] - ephemeris
tnt_binned_flux = tnt_binned_lc["Flux"]
tnt_binned_flux_err = tnt_binned_lc["Error"]

offset = 0.075

tom_flattened, tom_trend = flatten(tom_time, tom_flux, window_length=0.1, method='biweight', return_trend=True)
tnt_flattened, tnt_trend = flatten(tnt_time, tnt_flux, window_length=0.1, method='biweight', return_trend=True)
w1m_flattened, w1m_trend = flatten(w1m_time, w1m_flux, window_length=0.1, method='biweight', return_trend=True)

bin_size = 5 / 1440  # 5 minutes
w1m_binned_detrended_times = []
w1m_binned_detrended_relative_flux = []
w1m_binned_detrended_relative_flux_errs = []
for i in range(int(np.ceil(w1m_time[0] / bin_size)), int(np.ceil(w1m_time[-1] / bin_size))):
    mask = (w1m_time > i * bin_size) & (w1m_time < (i + 1) * bin_size)
    if np.sum(mask) < 5:
        continue
    w1m_binned_detrended_times.append(np.mean(w1m_time[mask]))
    w1m_binned_detrended_relative_flux.append(np.mean(w1m_flattened[mask]))
    w1m_binned_detrended_relative_flux_errs.append(np.sqrt(np.sum((w1m_flux_err / w1m_flattened)[mask] ** 2)) / np.sum(mask))

tnt_binned_detrended_times = []
tnt_binned_detrended_relative_flux = []
tnt_binned_detrended_relative_flux_errs = []
for i in range(int(np.ceil(tnt_time[0] / bin_size)), int(np.ceil(tnt_time[-1] / bin_size))):
    mask = (tnt_time > i * bin_size) & (tnt_time < (i + 1) * bin_size)
    if np.sum(mask) < 5:
        continue
    tnt_binned_detrended_times.append(np.mean(tnt_time[mask]))
    tnt_binned_detrended_relative_flux.append(np.mean(tnt_flattened[mask]))
    tnt_binned_detrended_relative_flux_errs.append(np.sqrt(np.sum((tnt_flux_err / tnt_flattened)[mask] ** 2)) / np.sum(mask))

tom_binned_detrended_times = []
tom_binned_detrended_relative_flux = []
tom_binned_detrended_relative_flux_errs = []
for i in range(int(np.ceil(tom_time[0] / bin_size)), int(np.ceil(tom_time[-1] / bin_size))):
    mask = (tom_time > i * bin_size) & (tom_time < (i + 1) * bin_size)
    if np.sum(mask) < 5:
        continue
    tom_binned_detrended_times.append(np.mean(tom_time[mask]))
    tom_binned_detrended_relative_flux.append(np.mean(tom_flattened[mask]))
    tom_binned_detrended_relative_flux_errs.append(np.sqrt(np.sum((tom_flux_err / tom_flattened)[mask] ** 2)) / np.sum(mask))

# plt.plot(tnt_binned_time / period % 1, tnt_binned_flux + np.floor(tnt_binned_time / period) * offset, '.')
# plt.plot(tnt_time / period % 1, tnt_trend + np.floor(tnt_time / period) * offset, 'r.', alpha=0.1)
# # plt.plot(tnt_time / period % 1, tnt_flux + np.floor(tnt_time / period) * offset, '.', alpha=0.1)
# # plt.plot(tnt_time / period % 1, tnt_flattened + np.floor(tnt_time / period) * offset, 'k.', alpha=0.1)
# plt.xlabel("T - T$_{0}$ (days)")
# plt.ylabel("Flux")
# plt.show()
# plt.plot(w1m_binned_time / period % 1, w1m_binned_flux + np.floor(w1m_binned_time / period) * offset, '.')
# plt.plot(w1m_time / period % 1, w1m_trend + np.floor(w1m_time / period) * offset, 'r.', alpha=0.1)
# # plt.plot(w1m_time / period % 1, w1m_flux + np.floor(w1m_time / period) * offset, '.', alpha=0.1)
# # plt.plot(w1m_time / period % 1, w1m_flattened + np.floor(w1m_time / period) * offset, 'k.', alpha=0.1)
# plt.xlabel("T - T$_{0}$ (days)")
# plt.ylabel("Flux")
# plt.show()
# plt.plot(tom_binned_time / period % 1, tom_binned_flux + np.floor(tom_binned_time / period) * offset, '.')
# plt.plot(tom_time / period % 1, tom_trend + np.floor(tom_time / period) * offset, 'r.', alpha=0.1)
# # plt.plot(tom_time / period % 1, tom_flux + np.floor(tom_time / period) * offset, '.', alpha=0.1)
# # plt.plot(tom_time / period % 1, tom_flattened + np.floor(tom_time / period) * offset, 'k.', alpha=0.1)
# plt.xlabel("T - T$_{0}$ (days)")
# plt.ylabel("Flux")
# plt.show()


blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
colours = [blue, orange, pink]

# plt.errorbar(w1m_time, w1m_flattened, yerr=w1m_flux_err / w1m_flattened, fmt='o', color=colours[0], alpha=0.05)
# plt.errorbar(tnt_time, tnt_flattened, yerr=tnt_flux_err / tnt_flattened, fmt='o', color=colours[1], alpha=0.025)
# plt.errorbar(tom_time, tom_flattened, yerr=tom_flux_err / tom_flattened, fmt='o', color=colours[2], alpha=0.05)
# plt.errorbar(w1m_binned_detrended_times, w1m_binned_detrended_relative_flux, yerr=w1m_binned_detrended_relative_flux_errs, fmt='o', label="W1m", color=colours[0])
# plt.errorbar(tnt_binned_detrended_times, tnt_binned_detrended_relative_flux, yerr=tnt_binned_detrended_relative_flux_errs, fmt='o', label="TNT", color=colours[1])
# plt.errorbar(tom_binned_detrended_times, tom_binned_detrended_relative_flux, yerr=tom_binned_detrended_relative_flux_errs, fmt='o', label="Tom", color=colours[2])

plt.plot(tom_time, tom_trend, 'r.', alpha=0.2)
plt.plot(w1m_time, w1m_trend, 'r.', alpha=0.2)
plt.plot(tnt_time, tnt_trend, 'r.', alpha=0.1)
plt.errorbar(w1m_time, w1m_flux, yerr=w1m_flux_err, fmt='o', color=colours[0], alpha=0.05)
plt.errorbar(tnt_time, tnt_flux, yerr=tnt_flux_err, fmt='o', color=colours[1], alpha=0.025)
plt.errorbar(tom_time, tom_flux, yerr=tom_flux_err, fmt='o', color=colours[2], alpha=0.05)
plt.errorbar(w1m_binned_time, w1m_binned_flux, yerr=w1m_binned_flux_err, fmt='o', color=colours[0], label="W1m")
plt.errorbar(tnt_binned_time, tnt_binned_flux, yerr=tnt_binned_flux_err, fmt='o', color=colours[1], label="TNT")
plt.errorbar(tom_binned_time, tom_binned_flux, yerr=tom_binned_flux_err, fmt='o', color=colours[2], label="Tom")


plt.xlim(43, 50)
plt.ylim(0.85, 1.15)
plt.xlabel("T - T$_{0}$ (days)")
plt.legend(loc="upper left")
plt.ylabel("Flux")
plt.show()