import numpy as np

plate_scale = 0.394  # arcsec/pixel  # 0.788 arcsec/pixel for 2x2 binning
red_gain = 2.011  # e-/ADU
blue_gain = 1.055  # e-/ADU
red_read_noise_electrons = 14.306  # e-  # 14.28 e- for 2x2 binning
blue_read_noise_electrons = 11.435  # e-  # 12.96 e- for 2x2 binning
red_dark_current = 0.9049  # e-/pixel/s   # 3.7867 e-/pixel/s for 2x2 binning
blue_dark_current = 0.0021  # e-/pixel/s   #  0.0243 e-/pixel/s for 2x2 binning


def bin_data(bin_size, time, flux, flux_err):
    binned_times = []
    binned_flux = []
    binned_flux_errs = []
    for i in range(int(np.ceil(time[0] / bin_size)), int(np.ceil(time[-1] / bin_size))):
        mask = (time > i * bin_size) & (time < (i + 1) * bin_size)
        if np.sum(mask) < 5:
            continue
        binned_times.append(np.mean(time[mask]))
        binned_flux.append(np.mean(flux[mask]))
        binned_flux_errs.append(
            np.sqrt(np.sum((flux_err / flux)[mask] ** 2)) / np.sum(mask))
    binned_times = np.array(binned_times)
    binned_flux = np.array(binned_flux)
    binned_flux_errs = np.array(binned_flux_errs)
    return binned_times, binned_flux, binned_flux_errs