from path import Path
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pickle
import warnings


def model_function(theta, t):
    m, c = theta[0], theta[1]
    theta = theta[2:]
    num_transits = len(theta) // 4
    model = m * t + c
    for i in range(num_transits):
        t_0 = theta[0 + i * 4]
        t_i = theta[1 + i * 4]
        t_e = theta[2 + i * 4]
        a = theta[3 + i * 4]
        model /= (a / (np.exp(-(t-t_0)/t_i) + np.exp((t-t_0)/t_e)) + 1)
    return model


warnings.filterwarnings("ignore", category=RuntimeWarning)
sampler_path = Path("/Users/nagro/PycharmProjects/w1m/mcmc/")

plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200


# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_detrended_flux.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["bjd"]
w1m_flux = w1m_lc["flux"]
w1m_flux_err = w1m_lc["err"]

# tom's data
tom_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_detrended_flux.csv")
tom_lc = np.genfromtxt(tom_lc_path, delimiter=",", names=True)
tom_time = tom_lc["bjd"]
tom_flux = tom_lc["flux"]
tom_flux_err = tom_lc["err"]

# tnt data
tnt_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_detrended_flux.csv")
tnt_lc = np.genfromtxt(tnt_lc_path, delimiter=",", names=True)
tnt_time = tnt_lc["bjd"]
tnt_flux = tnt_lc["flux"]
tnt_flux_err = tnt_lc["err"]

# gtc data
gtc_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_detrended_flux.csv")
gtc_lc = np.genfromtxt(gtc_lc_path, delimiter=",", names=True)
gtc_time = gtc_lc["bjd"]
gtc_flux = gtc_lc["flux"]
gtc_flux_err = gtc_lc["err"]

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
green = "#008000"
colours = [blue, orange, pink, green]

transit_path = Path("/Users/nagro/PycharmProjects/w1m/transit_params.csv")
transit_data = np.genfromtxt(transit_path, delimiter=",", names=True)


night = 3
obs = "tnt"  # change here
# change here
if obs == "w1m":
    flux_err_nights = np.split(w1m_flux_err, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
    flux_nights = np.split(w1m_flux, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
    time_nights = np.split(w1m_time, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
elif obs == "tom":
    flux_err_nights = np.split(tom_flux_err, np.where(np.diff(tom_time) > 0.5)[0] + 1)
    flux_nights = np.split(tom_flux, np.where(np.diff(tom_time) > 0.5)[0] + 1)
    time_nights = np.split(tom_time, np.where(np.diff(tom_time) > 0.5)[0] + 1)
elif obs == "tnt":
    flux_err_nights = np.split(tnt_flux_err, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    flux_nights = np.split(tnt_flux, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    time_nights = np.split(tnt_time, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
elif obs == "gtc":
    flux_err_nights = np.split(gtc_flux_err, np.where(np.diff(gtc_time) > 0.5)[0] + 1)
    flux_nights = np.split(gtc_flux, np.where(np.diff(gtc_time) > 0.5)[0] + 1)
    time_nights = np.split(gtc_time, np.where(np.diff(gtc_time) > 0.5)[0] + 1)

times = time_nights[night]
fluxes = flux_nights[night]
flux_err = flux_err_nights[night]

# get the transits with t_min in the range of the data
transit_data_nights = transit_data[
    (transit_data["t_min"] > np.min(times)) & (transit_data["t_min"] < np.max(times))
]

# build up theta from the transit data
theta = [0, 1]
for transit in transit_data_nights:
    t_0 = transit["t0"]
    t_i = transit["ti"]
    t_e = transit["te"]
    a = transit["a"]
    theta.extend([t_0, t_i, t_e, a])


# check also for the right telescope
transit_data_nights = transit_data_nights[transit_data_nights[obs] == 1]
print(transit_data_nights)
plt.errorbar(times - 2460000, fluxes, yerr=flux_err, fmt='o', color=colours[0], label=obs, alpha=0.5, zorder=1)
plt.plot(times - 2460000, model_function(theta, times), color=colours[1], label="Model", alpha=1)
for transit in transit_data_nights:
    t_min = transit["t_min"]
    depth = transit["depth"]
    identifier = transit["id"]
    plt.axvline(t_min - 2460000, color=colours[0], linestyle='--')
    plt.text(t_min - 2460000, 1.02, int(identifier), color="red", fontsize=26)

plt.xlabel("BJD - 2 460 000")
plt.title("SDSS1234")
plt.ylabel("Flux")
plt.legend()
plt.grid()
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.9)
plt.show()
exit()