from path import Path
import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten, transit_mask
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

# Load your data
lc_dir = Path("/Users/nagro/PycharmProjects/w1m/sdss1234_lightcurves")
tnt_loc = coord.EarthLocation(lat=18.590555 * u.deg, lon=98.486666 * u.deg, height=2457 * u.m)
gtc_loc = coord.EarthLocation(lat=28.75661 * u.deg, lon=-17.89203 * u.deg, height=2267 * u.m)
target_coord = coord.SkyCoord(ra=188.635570, dec=56.111952, unit="deg")  # SDSS1234+5606

# gtc data
lc_files = sorted(lc_dir.glob("*.asc"))
gtc_time = []
gtc_flux = []
gtc_flattened_flux = []
gtc_flux_err = []
gtc_flattened_flux_err = []

colours = ["violet", "blue", "green", "orange", "red"]
filters = ["u", "g", "r", "i", "z"]

for file, c, filter in zip(lc_files, colours, filters):
    gtc_time_i, gtc_flux_i, gtc_err_i = np.loadtxt(file, unpack=True, usecols=(0, 2, 3), skiprows=0)

    gtc_time_mjd = Time(gtc_time_i, format="mjd", scale="utc")
    gtc_time_jd = Time(gtc_time_mjd.jd, format="jd", scale="utc")
    ltt_bjd = gtc_time_jd.light_travel_time(target_coord, 'barycentric', location=gtc_loc)
    gtc_time_i = (gtc_time_jd.tdb + ltt_bjd).jd

    mask = transit_mask(
        time=gtc_time_i,
        period=14.803 / 24,
        duration=0.01,
        T0=2460824.491)

    gtc_flux_median = np.median(gtc_flux_i)
    gtc_flux_i /= gtc_flux_median
    gtc_err_i /= gtc_flux_median

    gtc_flux_flattened_i, gtc_trend_i = flatten(gtc_time_i, gtc_flux_i, window_length=0.02, method='cosine', return_trend=True, mask=mask)
    gtc_flat_err_i = gtc_err_i / gtc_trend_i

    plot = False
    if plot:
        plot_time = (gtc_time_i - 2460824.5) * 1440
        plt.errorbar(plot_time, gtc_flux_i, yerr=gtc_flat_err_i, fmt="o", color=c,
                     label="GTC White Light Curve", zorder=1, alpha=0.5)
        plt.plot(plot_time, gtc_trend_i, 'k-')
        plt.title(f"GTC {filter.upper()} Light Curve")
        plt.fill_between(plot_time, 0.5, 1.5, where=mask, color='gray', alpha=0.5, label='Transit Mask', zorder=0)
        # plt.xlim(-25, 0)
        plt.ylim(0.81, 1.15)
        plt.grid()
        plt.show()


        plt.errorbar(plot_time, gtc_flux_flattened_i, yerr=gtc_flat_err_i, fmt="o", color=c,
                     label="GTC White Light Curve Flattened", zorder=1, alpha=0.5)
        plt.xlabel("BJD - 2460824.5 (minutes)")
        plt.ylabel("Normalized Flux")
        plt.title(f"GTC {filter.upper()} Flattened Light Curve")
        plt.axhline(1.0, color='k', linestyle='--', label='Baseline')
        plt.ylim(0.81, 1.15)
        # plt.xlim(-25, 0)
        plt.grid()
        plt.show()
    print(f"{filter} - {np.sqrt(np.mean(gtc_flat_err_i**2)):.2%} photometric error")

    gtc_time.append(gtc_time_i)
    gtc_flux.append(gtc_flux_i)
    gtc_flux_err.append(gtc_err_i)
    gtc_flattened_flux.append(gtc_flux_flattened_i)
    gtc_flattened_flux_err.append(gtc_flat_err_i)

gtc_time = np.array(gtc_time).mean(axis=0)  # average time across all CCDs (in practice they are the same)
gtc_flux = np.array(gtc_flux)
gtc_flux_err = np.array(gtc_flux_err)
gtc_flattened_flux = np.array(gtc_flattened_flux)
gtc_flattened_flux_err = np.array(gtc_flattened_flux_err)

# create a white light curve by weighted averaging the fluxes from all CCDs
weights = 1 / gtc_flux_err**2
gtc_white_flux = np.sum(gtc_flux * weights, axis=0) / np.sum(weights, axis=0)
gtc_white_flux_err = np.sqrt(1 / np.sum(weights, axis=0))

gtc_white_flattened, gtc_white_trend = flatten(gtc_time, gtc_white_flux, window_length=0.01, method='cosine', return_trend=True, mask=mask)
gtc_white_detrended_flux_err = gtc_white_flux_err / gtc_white_trend

print(f"combined - {np.sqrt(np.mean(gtc_white_detrended_flux_err**2)):.2%} photometric error")

# combine the white light into a single array
gtc_flux = np.concatenate((gtc_white_flux[np.newaxis, :], gtc_flux), axis=0)
gtc_flux_err = np.concatenate((gtc_white_flux_err[np.newaxis, :], gtc_flux_err), axis=0)

gtc_flattened_flux = np.concatenate((gtc_white_flattened[np.newaxis, :], gtc_flattened_flux), axis=0)
gtc_flattened_flux_err = np.concatenate((gtc_white_detrended_flux_err[np.newaxis, :], gtc_flattened_flux_err), axis=0)


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
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/combined_lc_5.csv")
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

# plt.errorbar(w1m_time, w1m_flattened, yerr=w1m_detrended_flux_err, fmt="o", color="blue", label="W1m")
# plt.errorbar(tnt_time, tnt_flattened, yerr=tnt_detrended_flux_err, fmt="o", color="red", label="TNT")
# plt.errorbar(tom_time, tom_flattened, yerr=tom_detrended_flux_err, fmt="o", color="green", label="Tom")
# plt.xlabel("BJD")
# plt.ylabel("Normalized Flux")
# plt.title("Light Curves")
# plt.legend()
# plt.show()

# convert fluxes to mags
w1m_mag = -2.5 * np.log10(w1m_flux)
w1m_mag_err = np.abs(w1m_mag - (-2.5 * np.log10(w1m_flux + w1m_flux_err)))
w1m_detrended_mag = -2.5 * np.log10(w1m_flattened)
w1m_detrended_mag_err = np.abs(w1m_detrended_mag - (-2.5 * np.log10(w1m_flattened + w1m_detrended_flux_err)))

tnt_mag = -2.5 * np.log10(tnt_flux)
tnt_mag_err = np.abs(tnt_mag - (-2.5 * np.log10(tnt_flux + tnt_flux_err)))
tnt_detrended_mag = -2.5 * np.log10(tnt_flattened)
tnt_detrended_mag_err = np.abs(tnt_detrended_mag - (-2.5 * np.log10(tnt_flattened + tnt_detrended_flux_err)))

tom_mag = -2.5 * np.log10(tom_flux)
tom_mag_err = np.abs(tom_mag - (-2.5 * np.log10(tom_flux + tom_flux_err)))
tom_detrended_mag = -2.5 * np.log10(tom_flattened)
tom_detrended_mag_err = np.abs(tom_detrended_mag - (-2.5 * np.log10(tom_flattened + tom_detrended_flux_err)))

gtc_mag = -2.5 * np.log10(gtc_flux)
gtc_mag_err = np.abs(gtc_mag - (-2.5 * np.log10(gtc_flux + gtc_flux_err)))
gtc_detrended_mag = -2.5 * np.log10(gtc_flattened_flux)
gtc_detrended_mag_err = np.abs(gtc_detrended_mag - (-2.5 * np.log10(gtc_flattened_flux + gtc_flattened_flux_err)))

# save as csv files
gtc_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_mag.csv")
with open(gtc_mag_path, "w") as f:
    f.write("bjd,mag,err,mag_u,err_u,mag_g,err_g,mag_r,err_r,mag_i,err_i,mag_z,err_z\n")
    for i in range(len(gtc_time)):
        f.write(f"{gtc_time[i]:.8f},{gtc_mag[0, i]:.6f},{gtc_mag_err[0, i]:.6f},{gtc_mag[1, i]:.6f},{gtc_mag_err[1, i]:.6f},{gtc_mag[2, i]:.6f},{gtc_mag_err[2, i]:.6f},{gtc_mag[3, i]:.6f},{gtc_mag_err[3, i]:.6f},{gtc_mag[4, i]:.6f},{gtc_mag_err[4, i]:.6f},{gtc_mag[5, i]:.6f},{gtc_mag_err[5, i]:.6f}\n")

gtc_detr_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_detrended_mag.csv")
with open(gtc_detr_mag_path, "w") as f:
    f.write("bjd,mag,err,mag_u,err_u,mag_g,err_g,mag_r,err_r,mag_i,err_i,mag_z,err_z\n")
    for i in range(len(gtc_time)):
        f.write(f"{gtc_time[i]:.8f},{gtc_detrended_mag[0, i]:.6f},{gtc_detrended_mag_err[0, i]:.6f},{gtc_detrended_mag[1, i]:.6f},{gtc_detrended_mag_err[1, i]:.6f},{gtc_detrended_mag[2, i]:.6f},{gtc_detrended_mag_err[2, i]:.6f},{gtc_detrended_mag[3, i]:.6f},{gtc_detrended_mag_err[3, i]:.6f},{gtc_detrended_mag[4, i]:.6f},{gtc_detrended_mag_err[4, i]:.6f},{gtc_detrended_mag[5, i]:.6f},{gtc_detrended_mag_err[5, i]:.6f}\n")

gtc_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_flux.csv")
with open(gtc_flux_path, "w") as f:
    f.write("bjd,flux,err,flux_u,err_u,flux_g,err_g,flux_r,err_r,flux_i,err_i,flux_z,err_z\n")
    for i in range(len(gtc_time)):
        f.write(f"{gtc_time[i]:.8f},{gtc_flux[0, i]:.6f},{gtc_flux_err[0, i]:.6f},{gtc_flux[1, i]:.6f},{gtc_flux_err[1, i]:.6f},{gtc_flux[2, i]:.6f},{gtc_flux_err[2, i]:.6f},{gtc_flux[3, i]:.6f},{gtc_flux_err[3, i]:.6f},{gtc_flux[4, i]:.6f},{gtc_flux_err[4, i]:.6f},{gtc_flux[5, i]:.6f},{gtc_flux_err[5, i]:.6f}\n")

gtc_detr_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_detrended_flux.csv")
with open(gtc_detr_flux_path, "w") as f:
    f.write("bjd,flux,err,flux_u,err_u,flux_g,err_g,flux_r,err_r,flux_i,err_i,flux_z,err_z\n")
    for i in range(len(gtc_time)):
        f.write(f"{gtc_time[i]:.8f},{gtc_flattened_flux[0, i]:.6f},{gtc_flattened_flux_err[0, i]:.6f},{gtc_flattened_flux[1, i]:.6f},{gtc_flattened_flux_err[1, i]:.6f},{gtc_flattened_flux[2, i]:.6f},{gtc_flattened_flux_err[2, i]:.6f},{gtc_flattened_flux[3, i]:.6f},{gtc_flattened_flux_err[3, i]:.6f},{gtc_flattened_flux[4, i]:.6f},{gtc_flattened_flux_err[4, i]:.6f},{gtc_flattened_flux[5, i]:.6f},{gtc_flattened_flux_err[5, i]:.6f}\n")

w1m_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_mag.csv")
with open(w1m_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(w1m_time)):
        f.write(f"{w1m_time[i]:.8f},{w1m_mag[i]:.6f},{w1m_mag_err[i]:.6f}\n")

w1m_detr_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_detrended_mag.csv")
with open(w1m_detr_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(w1m_time)):
        f.write(f"{w1m_time[i]:.8f},{w1m_detrended_mag[i]:.6f},{w1m_detrended_mag_err[i]:.6f}\n")

w1m_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_flux.csv")
with open(w1m_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(w1m_time)):
        f.write(f"{w1m_time[i]:.8f},{w1m_flux[i]:.6f},{w1m_flux_err[i]:.6f}\n")

w1m_detr_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_detrended_flux.csv")
with open(w1m_detr_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(w1m_time)):
        f.write(f"{w1m_time[i]:.8f},{w1m_flattened[i]:.6f},{w1m_detrended_flux_err[i]:.6f}\n")

tnt_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_mag.csv")
with open(tnt_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(tnt_time)):
        f.write(f"{tnt_time[i]:.8f},{tnt_mag[i]:.6f},{tnt_mag_err[i]:.6f}\n")

tnt_detr_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_detrended_mag.csv")
with open(tnt_detr_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(tnt_time)):
        f.write(f"{tnt_time[i]:.8f},{tnt_detrended_mag[i]:.6f},{tnt_detrended_mag_err[i]:.6f}\n")

tnt_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_flux.csv")
with open(tnt_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(tnt_time)):
        f.write(f"{tnt_time[i]:.8f},{tnt_flux[i]:.6f},{tnt_flux_err[i]:.6f}\n")

tnt_detr_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_detrended_flux.csv")
with open(tnt_detr_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(tnt_time)):
        f.write(f"{tnt_time[i]:.8f},{tnt_flattened[i]:.6f},{tnt_detrended_flux_err[i]:.6f}\n")

tom_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_mag.csv")
with open(tom_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(tom_time)):
        f.write(f"{tom_time[i]:.8f},{tom_mag[i]:.6f},{tom_mag_err[i]:.6f}\n")

tom_detr_mag_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_detrended_mag.csv")
with open(tom_detr_mag_path, "w") as f:
    f.write("bjd,mag,err\n")
    for i in range(len(tom_time)):
        f.write(f"{tom_time[i]:.8f},{tom_detrended_mag[i]:.6f},{tom_detrended_mag_err[i]:.6f}\n")

tom_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_flux.csv")
with open(tom_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(tom_time)):
        f.write(f"{tom_time[i]:.8f},{tom_flux[i]:.6f},{tom_flux_err[i]:.6f}\n")

tom_detr_flux_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_detrended_flux.csv")
with open(tom_detr_flux_path, "w") as f:
    f.write("bjd,flux,err\n")
    for i in range(len(tom_time)):
        f.write(f"{tom_time[i]:.8f},{tom_flattened[i]:.6f},{tom_detrended_flux_err[i]:.6f}\n")