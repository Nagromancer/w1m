from path import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from wotan import flatten
from astropy.time import Time
from astropy import units as u
import astropy.coordinates as coord

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
tnt_loc = coord.EarthLocation(lat=18.590555 * u.deg, lon=98.486666 * u.deg, height=2457 * u.m)
target_coord = coord.SkyCoord(ra=188.635570, dec=56.111952, unit="deg")  # SDSS1234+5606

# tom's data
lc_files = sorted(lc_dir.glob("*tom.dat"))
tom_time = []
tom_flux = []
tom_flux_err = []

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

tom_time = np.array(tom_time)
tom_flux = np.array(tom_flux)
tom_flux_err = np.array(tom_flux_err)

# tnt data
lc_files = sorted(lc_dir.glob("*uspec_kg5.dat"))
tnt_time_jd = []
tnt_flux = []
tnt_flux_err = []

for file in lc_files:
    tnt_time_jd_i, mag_i, mag_err_i = np.loadtxt(file, unpack=True, usecols=(0, 2, 3), skiprows=0)
    tnt_flux_i = 10 ** (-0.4 * mag_i)
    tnt_flux_err_i = np.abs(tnt_flux_i - 10 ** (-0.4 * (mag_i + mag_err_i)))
    tnt_flux_median = np.median(tnt_flux_i)
    tnt_flux_i /= tnt_flux_median
    tnt_flux_err_i /= tnt_flux_median

    # non-outlier range is between 0.5 and 1.5
    outlier_mask = (0.5 < tnt_flux_i) & (tnt_flux_i < 1.5)
    tnt_flux_i = tnt_flux_i[outlier_mask]
    tnt_flux_err_i = tnt_flux_err_i[outlier_mask]
    tnt_time_jd_i = tnt_time_jd_i[outlier_mask]

    tnt_time_jd.extend(tnt_time_jd_i)
    tnt_flux.extend(tnt_flux_i)
    tnt_flux_err.extend(tnt_flux_err_i)

tnt_time_jd = np.array(tnt_time_jd)
tnt_flux = np.array(tnt_flux)
tnt_flux_err = np.array(tnt_flux_err)

tnt_outlier_mask = (0.5 < tnt_flux) & (tnt_flux < 1.5)
tnt_flux = tnt_flux[tnt_outlier_mask]
tnt_flux_err = tnt_flux_err[tnt_outlier_mask]
tnt_time_jd = tnt_time_jd[tnt_outlier_mask]

# tnt time is in JD in UTC scale - need to convert to BJD_TDB
tnt_time_jd = Time(tnt_time_jd, format="jd", scale="utc")
ltt_bjd = tnt_time_jd.light_travel_time(target_coord, 'barycentric', location=tnt_loc)
tnt_time = (tnt_time_jd.tdb + ltt_bjd).jd


# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/combined_lc_5.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)

w1m_time = w1m_lc["Time_BJD"]
w1m_flux = w1m_lc["Flux"]
w1m_flux_err = w1m_lc["Error"]

# non-outlier range is between 0.5 and 1.5
outlier_mask = (0.75 < w1m_flux) & (w1m_flux < 1.25)
w1m_flux = w1m_flux[outlier_mask]
w1m_flux_err = w1m_flux_err[outlier_mask]
w1m_time = w1m_time[outlier_mask]


tom_flattened, tom_trend = flatten(tom_time, tom_flux, window_length=0.1, method='biweight', return_trend=True)
tom_detrended_flux_err = tom_flux_err / tom_trend
tnt_flattened, tnt_trend = flatten(tnt_time, tnt_flux, window_length=0.1, method='biweight', return_trend=True)
tnt_detrended_flux_err = tnt_flux_err / tnt_trend
w1m_flattened, w1m_trend = flatten(w1m_time, w1m_flux, window_length=0.1, method='biweight', return_trend=True)
w1m_detrended_flux_err = w1m_flux_err / w1m_trend

plt.errorbar(w1m_time, w1m_flattened, yerr=w1m_detrended_flux_err, fmt="o", color="blue", label="W1m")
plt.errorbar(tnt_time, tnt_flattened, yerr=tnt_detrended_flux_err, fmt="o", color="red", label="TNT")
plt.errorbar(tom_time, tom_flattened, yerr=tom_detrended_flux_err, fmt="o", color="green", label="Tom")
plt.xlabel("BJD")
plt.ylabel("Normalized Flux")
plt.title("Light Curves")
plt.legend()
plt.show()

# save the detrended data
# save the combined light curve as a csv
combined_lc = Table()
combined_lc["Time_BJD"] = tom_time
combined_lc["Flux"] = tom_flattened
combined_lc["Error"] = tom_detrended_flux_err
combined_lc.write(f"detrended_tom.csv", overwrite=True)

combined_lc = Table()
combined_lc["Time_BJD"] = tnt_time
combined_lc["Flux"] = tnt_flattened
combined_lc["Error"] = tnt_detrended_flux_err
combined_lc.write(f"detrended_tnt.csv", overwrite=True)

combined_lc = Table()
combined_lc["Time_BJD"] = w1m_time
combined_lc["Flux"] = w1m_flattened
combined_lc["Error"] = w1m_detrended_flux_err
combined_lc.write(f"detrended_w1m.csv", overwrite=True)
