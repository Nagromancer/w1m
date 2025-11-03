import numpy as np
import emcee
from path import Path
import corner
import warnings
import matplotlib.pyplot as plt
import pickle

from integrate_sec import integrate

warnings.filterwarnings("ignore", category=RuntimeWarning)
sampler_path = Path("/Users/nagro/PycharmProjects/w1m/multi/")

np.random.seed(42)

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
green = "#008000"
colours = [blue, orange, pink, green]

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

labels = ["$m$", "$c$", "$t\mathdefault{_0}$", "$t\mathdefault{_{i,0}}$", "$t\mathdefault{_{e,0}}$",
          "$a\mathdefault{_0}$", "$t\mathdefault{_1}$", "$t\mathdefault{_{i,1}}$", "$t\mathdefault{_{e,1}}$",
          "$a\mathdefault{_1}$", "$t\mathdefault{_2}$", "$t\mathdefault{_{i,2}}$", "$t\mathdefault{_{e,2}}$",
          "$a\mathdefault{_2}$", "$t\mathdefault{_3}$", "$t\mathdefault{_{i,3}}$", "$t\mathdefault{_{e,3}}$",
          "$a\mathdefault{_3}$", "$t\mathdefault{_4}$", "$t\mathdefault{_{i,4}}$", "$t\mathdefault{_{e,4}}$",
          "$a\mathdefault{_4}$"]


def model_function(theta, t):
    """
    theta = [m, c, t0_1, ti_1, te_1, a_1, t0_2, ti_2, te_2, a_2, ...]
    """
    m, c = theta[0], theta[1]
    params = theta[2:]
    num_transits = len(params) // 4

    model = m * t + c
    for i in range(num_transits):
        t0, ti, te, a = params[4 * i: 4 * (i + 1)]
        model /= (a / (np.exp(-(t - t0) / ti) + np.exp((t - t0) / te)) + 1)
    return model


def log_likelihood(theta, x, y, yerr):
    model = model_function(theta, x)
    return -0.5 * np.sum(((y - model) ** 2 / yerr ** 2) + np.log(2 * np.pi * yerr ** 2))


def log_prior(theta):
    m, c = theta[0], theta[1]
    params = theta[2:]
    num_transits = len(params) // 4
    t0s, tis, tes, amps = params[0::4], params[1::4], params[2::4], params[3::4]

    # Flat, bounded priors (modify as physically appropriate)
    if np.any(tis <= 0) or np.any(tes <= 0):
        return -np.inf
    if np.any(tis > 0.02) or np.any(tes > 0.04):
        return -np.inf
    if np.any(amps <= 0) or np.any(amps > 0.5):
        return -np.inf

    return 0.0  # flat within bounds


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


def fit_model(x, y, yerr, initial, t0, n_components, obs, night, transit, idx, nwalkers=50, nsteps=3000, burnin=1000):
    ndim = 2 + 4 * n_components  # m, c + (t0, ti, te, a)*n
    labels_i = labels[:ndim]

    if obs == "w1m":
        obs_flag = "1,0,0,0"
    elif obs == "tom":
        obs_flag = "0,1,0,0"
    elif obs == "tnt":
        obs_flag = "0,0,1,0"
    elif obs == "gtc":
        obs_flag = "0,0,0,1"

    mask = ((x - t0) < 0.05) & ((x - t0) > -0.05)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    initial = initial[:ndim]

    # plt.errorbar(x + ephemeris - 2460000, y, yerr=yerr, fmt='o', color=colours[2], label="Data", alpha=0.5, zorder=1)
    # plt.plot(x, model_function(initial, x), 'g-', label="Initial Guess")
    # plt.xlabel("Time (BJD - 2,460,000)")
    # plt.ylabel("Flux")
    # plt.title(f"Initial Model Guess - Transit {idx} - {n_components} Component(s)")
    # plt.legend()
    # plt.grid()
    # plt.show()

    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    full_sampler_path = sampler_path / f'sampler_{obs}_{night}_{transit}_{n_components}.pkl'
    if not Path(full_sampler_path).exists():
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, nsteps, progress=True)

        # pickle the sampler
        with open(full_sampler_path, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(full_sampler_path, 'rb') as f:
            sampler = pickle.load(f)

    flat_samples = sampler.get_chain(discard=burnin, flat=True)

    integral_samples = flat_samples[:, 2:]
    area_centroid = np.zeros((integral_samples.shape[0], 2))
    for i in range(integral_samples.shape[0]):
        area, centroid = integrate(integral_samples[i])
        area_centroid[i, 0] = area
        area_centroid[i, 1] = centroid
    area_chain = area_centroid[:, 0]
    centroid_chain = area_centroid[:, 1]

    area_mcmc = np.percentile(area_chain, [16, 50, 84])
    area_minus, area_plus = np.diff(area_mcmc)
    area_best = area_mcmc[1]

    centroid_mcmc = np.percentile(centroid_chain, [16, 50, 84])
    centroid_minus, centroid_plus = np.diff(centroid_mcmc)
    centroid_best = centroid_mcmc[1]
    print(f"{transit},{obs_flag},{n_components},", end="")
    print(f"{centroid_best + ephemeris:.6f},{centroid_minus:.6f},{centroid_plus:.6f},", end="")
    print(f"{area_best:.6f},{area_minus:.6f},{area_plus:.6f},", end="")

    best_fits = []
    lower_uncertainties = []
    upper_uncertainties = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fits.append(mcmc[1])
        lower_uncertainties.append(q[0])
        upper_uncertainties.append(q[1])

    for i in range(n_components):
        print(
            f"{best_fits[4 * i + 2] + ephemeris:.6f},{lower_uncertainties[4 * i + 2]:.6f},{upper_uncertainties[4 * i + 2]:.6f}",
            end=",")
        print(f"{best_fits[4 * i + 3]:.6f},{lower_uncertainties[4 * i + 3]:.6f},{upper_uncertainties[4 * i + 3]:.6f}",
              end=",")
        print(f"{best_fits[4 * i + 4]:.6f},{lower_uncertainties[4 * i + 4]:.6f},{upper_uncertainties[4 * i + 4]:.6f}",
              end=",")
        print(
            f"{best_fits[4 * i + 5]:.6f},{lower_uncertainties[4 * i + 5]:.6f},{upper_uncertainties[4 * i + 5]:.6f},",
            end="")
    print("")

    log_probs = sampler.get_log_prob(discard=burnin, flat=True)
    best_idx = np.argmax(log_probs)
    best_theta = flat_samples[best_idx]

    samples = sampler.get_chain(discard=burnin)
    plot = False
    if plot:
        # fig, axes = plt.subplots(ndim, figsize=(15, 2 * ndim), sharex=True)
        # for i in range(ndim):
        #     ax = axes[i]
        #     ax.plot(samples[:, :, i], "k", alpha=0.3)
        #     ax.set_xlim(0, len(samples))
        #     ax.set_ylabel(labels[i])
        #
        # axes[-1].set_xlabel("Step Number")
        # plt.tight_layout()
        # plt.show()

        plt.rcParams["font.size"] = 16
        fig = corner.corner(
            flat_samples, labels=labels_i
        )
        fig.suptitle(f"Posterior Distributions - {n_components} Component(s)", fontsize=20)
        plt.show()
        plt.close()
        plt.rcParams["font.size"] = 32

    plt.errorbar(x + ephemeris - 2460000, y, yerr=yerr, fmt='o', color=colours[2], label="Data", alpha=0.5, zorder=1)
    plt.plot(x + ephemeris - 2460000, model_function(best_theta, x), 'r-', label="Best Fit")
    plt.xlabel("Time (BJD - 2,460,000)")
    plt.ylabel("Normalized Flux")
    plt.title(f"Best Fit Model - Transit {transit} ($i={idx}$) - {n_components} Component(s)")
    # vline at centroid
    plt.axvline(centroid_best + ephemeris - 2460000, color='g', linestyle='--', label="Centroid")
    plt.legend()
    plt.grid()
    plt.show()

    # # plot model residuals
    # plt.errorbar(x + ephemeris - 2460000, y - model_function(best_theta, x), yerr=yerr, fmt='o', color=colours[2], alpha=0.5, zorder=1)
    # plt.axhline(0, color='r', linestyle='--')
    # plt.xlabel("Time (BJD - 2,460,000)")
    # plt.ylabel("Residuals")
    # plt.title(f"Model Residuals - Transit {idx} - {n_components} Component(s)")
    # plt.grid()
    # plt.show()

    # print("(", end="")
    # print(*np.round(best_theta, 4), sep=", ", end=")\n")

    return {
        "sampler": sampler,
        "samples": samples,
        "log_probs": log_probs,
        "best_theta": best_theta,
        "logL_max": log_likelihood(best_theta, x, y, yerr),
        "k": ndim,
        "n": len(x)
    }


def model_selection_stats(fit):
    n, k, logL = fit["n"], fit["k"], fit["logL_max"]
    AIC = 2 * k - 2 * logL
    AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    BIC = k * np.log(n) - 2 * logL
    return {"AIC": AIC, "AICc": AICc, "BIC": BIC}


ephemeris = 2460710.3964036

# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_detrended_flux.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["bjd"] - ephemeris
w1m_flux = w1m_lc["flux"]
w1m_flux_err = w1m_lc["err"]

# tom's data
tom_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tom_detrended_flux.csv")
tom_lc = np.genfromtxt(tom_lc_path, delimiter=",", names=True)
tom_time = tom_lc["bjd"] - ephemeris
tom_flux = tom_lc["flux"]
tom_flux_err = tom_lc["err"]

# tnt data
tnt_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/tnt_detrended_flux.csv")
tnt_lc = np.genfromtxt(tnt_lc_path, delimiter=",", names=True)
tnt_time = tnt_lc["bjd"] - ephemeris
tnt_flux = tnt_lc["flux"]
tnt_flux_err = tnt_lc["err"]

# gtc data
gtc_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/gtc_detrended_flux.csv")
gtc_lc = np.genfromtxt(gtc_lc_path, delimiter=",", names=True)
gtc_time = gtc_lc["bjd"] - ephemeris
gtc_flux = gtc_lc["flux"]
gtc_flux_err = gtc_lc["err"]

initials = [(0.1995, 1.007, -0.0027, 0.0004, 0.0116, 0.1359, 0.0079, 0.0038, 0.0018, 0.1866),  # 0
            (-0.4929, 1.9401, 1.8482, 0.0004, 0.0114, 0.1677, 1.8579, 0.0012, 0.0024, 0.2477),  # 1
            (-0.1637, 1.7128, 4.3164, 0.0004, 0.0106, 0.173, 4.3275, 0.0016, 0.0008, 0.2669),  # 2
            (-0.1176, 2.1872, 9.8781, 0.0048, 0.0073, 0.2919),  # 3
            (0.1753, -2.1239, 17.9026, 0.0045, 0.0014, 0.1233, 17.891, 0.0006, 0.011, 0.1371),  # 4
            (0.3089, -4.8933, 19.1296, 0.0027, 0.0053, 0.3417),  # 5
            (0.5266, -9.7015, 20.3704, 0.0037, 0.0004, 0.1623, 20.3595, 0.0014, 0.0058, 0.1563),  # 6
            (-0.4288, 9.7518, 20.3704, 0.0026, 0.0004, 0.2547, 20.3585, 0.0008, 0.0099, 0.1589),  # 7
            (0.1645, -2.6474, 22.2209, 0.0038, 0.0021, 0.1963, 22.2095, 0.0005, 0.007, 0.1164),  # 8
            (0.0514, -0.4134, 27.7733, 0.0052, 0.0022, 0.1717, 27.7618, 0.0, 0.0059, 0.1004),  # 9
            (0.6984, -19.2332, 28.9958, 0.0003, 0.0185, 0.1078, 29.0088, 0.0055, 0.0006, 0.1068),  # 10
            (0.2894, -9.1417, 35.1745, 0.0012, 0.0013, 0.1674),  # 11
            (-0.3343, 15.8617, 44.4181, 0.0009, 0.0065, 0.1938, 44.4287, 0.0014, 0.0004, 0.2183),  # 12
            (-2.1735, 101.6229, 46.2687, 0.0007, 0.0062, 0.2238),  # 13
            (-0.0268, 2.25, 46.2701, 0.0023, 0.0093, 0.1933),  # 14
            (0.3909, -17.5463, 47.5014, 0.0001, 0.0239, 0.1144),  # 15
            (-0.3078, 16.201, 49.3551, 0.0016, 0.0041, 0.1917, 49.3627, 0.0009, 0.0001, 0.1982),  # 16
            (0.3021, -16.1317, 56.7548, 0.0002, 0.0106, 0.1644),  # 17
            (-0.1684, 11.1004, 59.8419, 0.0041, 0.0045, 0.2315),  # 18
            (-0.0238, 2.7679, 74.0225, 0.0009, 0.0045, 0.1632),  # 19
            (0.0196, -0.4771, 75.2545, 0.0005, 0.0036, 0.1053),  # 20
            (0.013, -0.039, 80.1898, 0.0016, 0.0028, 0.1495),  # 21
            (-0.0217, 2.7804, 82.0373, 0.0006, 0.0043, 0.1234),  # 22
            (-0.0437, 4.9395, 90.0542, 0.0018, 0.0024, 0.1101),  # 23
            (-0.0215, 3.2778, 106.0794, 0.0011, 0.0024, 0.1128),  # 24
            (0.0016, 0.8144, 114.0917, 0.0001, 0.0017, 0.0371, 114.0948, 0.0013, 0.0004, 0.0695, 114.0966, 0.0005,
             0.001, 0.0235),  # , 114.093, 0.0003, 0.0002, 0.0242),  # 25
            (-0.0417, 1.097, 2.1717, 0.0007, 0.008, 0.205),  # 26
            (1.1808, -3.761, 4.0271, 0.0003, 0.0069, 0.1301),  # 27
            (0.0, 1, 5.2649, 0.0007, 0.0068, 0.1641),  # 28
            (0.0, 1, 10.2116, 0.0004, 0.0062, 0.1642),  # 29
            (0.0, 1, 13.3027, 0.0001, 0.0069, 0.1395),  # 30
            (-0.0101, 1.1539, 15.1593, 0.0012, 0.0035, 0.135),  # 31
            (0.0, 1, 18.2486, 0.0007, 0.0038, 0.161),  # 32
            (-0.0648, 2.3921, 21.3403, 0.0012, 0.0035, 0.1938),  # 33
            (-0.0048, 1.1127, 23.1957, 0.0017, 0.0011, 0.1732),  # 34
            (0.0756, 1.0056, -0.0237, 0.0004, 0.0015, 0.1169),  # 35
            (0.0294, 0.9129, 3.0559, 0.0003, 0.0024, 0.0736),  # 36
            (0.0, 1, 1.9611, 0.0012, 0.0042, 0.2215),  # 37
            (0.2087, 0.3461, 3.1951, 0.001, 0.0054, 0.2412),  # 38
            (0.0, 1, 9.9819, 0.001, 0.006, 0.1522),  # 39
            (0.003, 0.9538, 16.1507, 0.0005, 0.004, 0.1808),  # 40
            (0.0, 1, 17.385, 0.0015, 0.0031, 0.2442),  # 41
            (0.0, 1, 18.0016, 0.0007, 0.0026, 0.2675),  # 42
            (0.0, 1, 19.2349, 0.0006, 0.0033, 0.2684),  # 43
            (0.0, 1, 20.4689, 0.001, 0.0029, 0.3111),  # 44
            (-0.0191, 1.4053, 21.0842, 0.0009, 0.0052, 0.2208),  # 45
            (0.0, 1, 22.3204, 0.002, 0.0018, 0.1517),  # 46
            (0.0, 1, 27.8737, 0.0044, 0.0029, 0.1, 27.8737, 0.0044, 0.0029, 0.1),  # 47
            (0.0, 1, 1.9843, 0.0005, 0.0016, 0.1608),  # 48
            (0.0, 1, 3.2169, 0.0005, 0.0023, 0.1774),  # 49
            (0.0, 1, 2.0197, 0.0003, 0.0064, 0.05, 2.0197, 0.0003, 0.0064, 0.05),  # 50
            (-0.0032, 1.0227, 5.1694, 0.0003, 0.006, 0.22),  # 51
            (-0.0696, 1.7191, 10.105, 0.0022, 0.0083, 0.1732),  # 52
            (0.0221, 0.7101, 13.192, 0.0006, 0.0016, 0.278),  # 53
            (0.0, 1, 16.2759, 0.0003, 0.0047, 0.1605),  # 54
            (-0.0019, 1.0367, 18.1269, 0.0005, 0.0035, 0.1825),  # 55
            (0.0, 1, 19.3612, 0.0007, 0.0038, 0.1446),  # 56
            (0.0, 1, 21.2111, 0.0003, 0.0019, 0.1664),  # 57
            (-0.007, 1.1887, 26.7618, 0.0002, 0.0024, 0.1497),  # 58
            (0.0, 1, 27.9965, 0.0011, 0.0015, 0.1539),  # 59
            ]

bad_indices = [11, 15, 17, 20, 52]

transits = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 7,
            7, 7, 7, 7, 7, 7, 7, 7]
t0s = [0.005, 1.860, 4.328, 9.88, 17.9, 19.13, 20.37, 20.365, 22.2175, 27.77, 29.005,
       35.175, 44.425, 46.27, 46.28, 47.5125, 49.36, 56.76, 59.84, 74.025, 75.26, 80.19,
       82.04, 90.06, 106.08, 114.094, 2.18, 4.03, 5.27, 10.215, 13.31, 15.165, 18.255,
       21.345, 23.2, -0.02, 3.06, 1.965, 3.2, 9.985, 16.155, 17.39, 18.005, 19.24, 20.46,
       21.09, 22.325, 27.88, 1.99, 3.22, 2.025, 5.175, 10.11, 13.2, 16.28, 18.13, 19.37,
       21.215, 26.77, 28.0]
obss = ["tnt", "tnt", "w1m", "tnt", "tnt", "w1m", "w1m", "tom", "w1m", "tnt", "tnt", "w1m", "tom",
        "tom", "w1m", "tom", "tom", "tnt", "tnt", "w1m", "w1m", "w1m", "w1m", "w1m", "w1m", "gtc",
        "w1m", "tnt", "w1m", "w1m", "w1m", "w1m", "w1m", "tom", "w1m", "tnt", "tnt", "tnt", "w1m",
        "tnt", "w1m", "w1m", "tnt", "w1m", "tom", "w1m", "w1m", "tnt", "tnt", "w1m", "tnt", "w1m",
        "w1m", "w1m", "w1m", "w1m", "w1m", "w1m", "tnt", "tnt"]
nights = [0, 1, 2, 4, 5, 10, 11, 0, 13, 7, 8, 15, 2, 3, 19, 4, 6, 11, 12, 21, 22, 26, 28, 31, 36,
          0, 0, 3, 3, 4, 5, 6, 9, 1, 14, 0, 2, 1, 1, 4, 7, 8, 5, 10, 0, 12, 13, 7, 1, 1, 1, 3, 4,
          5, 7, 9, 10, 12, 6, 7]

start_idx = 0
end_idx = None

sec_parameters = [label.replace("\mathdefault{", "").replace("}", "").replace("{", "").replace(",", "_")[1:-1] for label in labels[2:-8]]
print("id,w1m,rvo,tnt,gtc,n_comp,t_c,t_c_minus,t_c_plus,area,area_minus,area_plus,", end="")
print(",".join([f"{param},{param}_minus,{param}_plus" for param in sec_parameters]))
for i, initial, t0, obs, night, transit in zip(range(len(initials))[start_idx:end_idx], initials[start_idx:end_idx],
                                               t0s[start_idx:end_idx], obss[start_idx:end_idx],
                                               nights[start_idx:end_idx], transits[start_idx:end_idx]):
    if i in bad_indices:
        continue
    if obs == "tnt":
        flux_err_nights = np.split(tnt_flux_err, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
        flux_nights = np.split(tnt_flux, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
        time_nights = np.split(tnt_time, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    if obs == "gtc":
        flux_err_nights = np.split(gtc_flux_err, np.where(np.diff(gtc_time) > 0.5)[0] + 1)
        flux_nights = np.split(gtc_flux, np.where(np.diff(gtc_time) > 0.5)[0] + 1)
        time_nights = np.split(gtc_time, np.where(np.diff(gtc_time) > 0.5)[0] + 1)
    if obs == "w1m":
        flux_err_nights = np.split(w1m_flux_err, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
        flux_nights = np.split(w1m_flux, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
        time_nights = np.split(w1m_time, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
    if obs == "tom":
        flux_err_nights = np.split(tom_flux_err, np.where(np.diff(tom_time) > 0.5)[0] + 1)
        flux_nights = np.split(tom_flux, np.where(np.diff(tom_time) > 0.5)[0] + 1)
        time_nights = np.split(tom_time, np.where(np.diff(tom_time) > 0.5)[0] + 1)

    time = time_nights[night]
    flux = flux_nights[night]
    flux_err = flux_err_nights[night]

    fits = {}
    stats = {}

    single_run = True
    if not single_run:
        for ncomp in range(1, (len(initial) - 2) // 4 + 1):
            print(f"Fitting model with {ncomp} components...")
            fit = fit_model(time, flux, flux_err, initial, t0, ncomp, obs, night, transit, i, nsteps=3000 * ncomp,
                            burnin=3000 * (ncomp - 1) + 1000)
            fits[ncomp] = fit
            stats[ncomp] = model_selection_stats(fit)
        for ncomp, s in stats.items():
            print(f"{ncomp} comps: AIC={s['AIC']:.1f}, AICc={s['AICc']:.1f}, BIC={s['BIC']:.1f}")

        # print the delta BIC relative to the first model
        first_bic = stats[1]['BIC']

        print("\nModel Comparison:")
        for ncomp, s in stats.items():
            delta_bic = s['BIC'] - first_bic
            print(f"{ncomp} comps: ΔBIC={delta_bic:.1f}")
    else:
        # single mode
        ncomp = (len(initial) - 2) // 4
        # print(f"Fitting model with {ncomp} components...")
        fit = fit_model(time, flux, flux_err, initial, t0, ncomp, obs, night, transit, i, nsteps=3000 * ncomp,
                        burnin=3000 * (ncomp - 1) + 1000)
        fits[ncomp] = fit
        stats[ncomp] = model_selection_stats(fit)
        s = stats.items()
