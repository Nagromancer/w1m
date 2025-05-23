from path import Path
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pickle
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sampler_path = Path("/Users/nagro/PycharmProjects/w1m/mcmc/")

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


def log_likelihood(theta, x, y, yerr):
    model = model_function(theta, x)
    return -0.5 * np.sum(((y - model) ** 2 / yerr ** 2) + np.log(2 * np.pi * yerr ** 2))


def log_prior(theta):
    m = theta[0]
    c = theta[1]
    theta = theta[1:]

    t_0s = theta[1::4]
    t_is = theta[2:][::4]
    t_es = theta[3:][::4]
    amps = theta[4:][::4]
    # if np.abs(c - 1) > 0.1:
    #     return -np.inf
    if np.any(theta[1:] < -0.1):
        return -np.inf
    if np.any(t_is > 0.02):
        return -np.inf
    if np.any(t_es > 0.04):
        return -np.inf
    if np.any(amps > 0.5):
        return -np.inf
    return 0.0 # Flat prior


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


def integrate(a, t_i, t_e):
    # a, t_i, t_e are numpy arrays
    if np.isscalar(a) or np.isscalar(t_i) or np.isscalar(t_e):
        raise ValueError("a, t_i, and t_e must be numpy arrays")

    t_max = 15 * t_e
    t_min = -15 * t_i
    t = np.linspace(t_min, t_max, 1000)
    # 0.009020, 0.00042, 0.00043
    # evaluate the function at these times
    f = 1 - 1 / (a / (np.exp(-t / t_i) + np.exp(t / t_e)) + 1)

    # integrate the function
    integral = np.trapz(f, t, axis=0)

    return integral


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

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

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
colours = [blue, orange, pink]


initial_poses = [(-0.0417, 1.097, 2.1717, 0.0007, 0.008, 0.205),
                 (0.2087, 0.3461, 3.1951, 0.001, 0.0054, 0.2412, 3.2169, 0.0005, 0.0023, 0.1774),
                 (0.031, 0.8707, 4.3216, 0.0031, 0.0067, 0.3097),
                 (-0.0032, 1.0227, 5.1694, 0.0003, 0.006, 0.22, 5.2649, 0.0007, 0.0068, 0.1641),
                 (-0.0696, 1.7191, 10.105, 0.0022, 0.0083, 0.1732, 10.2116, 0.0004, 0.0062, 0.1642),
                 (0.0221, 0.7101, 13.192, 0.0006, 0.0016, 0.278, 13.3027, 0.0001, 0.0069, 0.1395),
                 (-0.0101, 1.1539, 15.1593, 0.0012, 0.0035, 0.135),
                 (0.003, 0.9538, 16.1507, 0.0005, 0.004, 0.1808, 16.2759, 0.0003, 0.0047, 0.1605),
                 (0.0193, 0.6693, 17.2783, 0.0024, 0.0043, 0.4206, 17.385, 0.0015, 0.0031, 0.2442),
                 (-0.0019, 1.0367, 18.1269, 0.0005, 0.0035, 0.1825, 18.2486, 0.0007, 0.0038, 0.161),
                 (0.0148, 0.7184, 19.1299, 0.0024, 0.0044, 0.3223, 19.2349, 0.0006, 0.0033, 0.2684, 19.3612, 0.0007, 0.0038, 0.1446),
                 (0.1682, -2.4131, 20.3702, 0.0073, 0.0009, 0.1855),
                 (-0.0191, 1.4053, 21.0842, 0.0009, 0.0052, 0.2208, 21.2111, 0.0003, 0.0019, 0.1664),
                 (0.0059, 0.871, 22.2195, 0.0054, 0.0029, 0.2459, 22.3204, 0.002, 0.0018, 0.3017),
                 (-0.0048, 1.1127, 23.1957, 0.0017, 0.0011, 0.1732),
                 (0.3508, -11.3103, 35.1649, 0.0013, 0.0212, 0.1446),
                 (1,),
                 (1,),
                 (1,),
                 (0.0369, -0.7045, 46.2704, 0.0024, 0.0086, 0.1871),
                 (1,),
                 (-0.0238, 2.7679, 74.0225, 0.0009, 0.0045, 0.1632),
                 (0.0196, -0.4771, 75.2545, 0.0005, 0.0036, 0.1053),
                 (1,),
                 (1,),
                 (1,),
                 (0.013, -0.039, 80.1898, 0.0016, 0.0028, 0.1495),
                 (1,),
                 (-0.0217, 2.7804, 82.0373, 0.0006, 0.0043, 0.1234),
                 (1,),
                 (1,),
                 (-0.0437, 4.9395, 90.0542, 0.0018, 0.0024, 0.1101),
                 ]
initial_poses = [(0.0756, 1.0056, -0.0237, 0.0004, 0.0015, 0.1169, 0.0015, 0.0028, 0.0067, 0.3038),
                 (-0.047, 1.1013, 1.8484, 0.0004, 0.0157, 0.1562, 1.9611, 0.0012, 0.0042, 0.2215, 1.9843, 0.0005, 0.0016, 0.1608, 2.0197, 0.0003, 0.0064, 0.1058),
                 (0.0294, 0.9129, 3.0559, 0.0003, 0.0024, 0.0736),
                 (1.1808, -3.761, 4.0271, 0.0003, 0.0069, 0.1301),
                 (-0.1178, 2.1853, 9.8787, 0.005, 0.0066, 0.2821, 9.9819, 0.001, 0.006, 0.1522),
                 (-0.0179, 1.3276, 17.8957, 0.0031, 0.0059, 0.264, 18.0016, 0.0007, 0.0026, 0.2675),
                 (-0.007, 1.1887, 26.7618, 0.0002, 0.0024, 0.1497),
                 (-0.0471, 2.3194, 27.7706, 0.0055, 0.0035, 0.2338, 27.8737, 0.0044, 0.0029, 0.1993, 27.9965, 0.0011, 0.0015, 0.1539),
                 (0.0419, -0.2091, 29.0, 0.0024, 0.0062, 0.2237),
                 (1,),
                 (1,),
                 (-0.2458, 14.9733, 56.7549, 0.0007, 0.0089, 0.1977),
                 (0, 1, 59.84, 0.0007, 0.0089, 0.1977),
                 ]
# initial_poses = [(-0.1228, 3.5104, 20.3706, 0.0061, 0.0006, 0.2812, 20.4689, 0.001, 0.0029, 0.3111),
#                  (-0.0648, 2.3921, 21.3403, 0.0012, 0.0035, 0.1938),
#                  (-0.0135, 1.6024, 44.4185, 0.0009, 0.0082, 0.1765),
#                  (-0.1883, 9.7497, 46.2679, 0.0002, 0.0394, 0.0815),
#                  (-0.2032, 10.676, 47.5039, 0.0022, 0.0102, 0.1594),
#                  (1,),
#                  (-0.011, 1.5462, 49.3553, 0.0016, 0.005, 0.1807),
#                  ]

print("t_min, t_min_-, t_min_+, depth, depth_-, depth_+, width, width_-, width+, area, area_-, area_+, t0, t0_-, t0_+, ti, ti_-, ti_+, te, te_-, te_+, a, a_-, a_+, w1m, tom, tnt")
for night, initial_pos in enumerate(initial_poses):
    if len(initial_pos) == 1:
        continue

    num_parameters = len(initial_pos)
    pos = initial_pos + 1e-5 * np.random.randn(48, num_parameters)

    tnt_flux_err_nights = np.split(tnt_flux_err, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    tnt_flux_nights = np.split(tnt_flux, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    tnt_time_nights = np.split(tnt_time, np.where(np.diff(tnt_time) > 0.5)[0] + 1)
    times = tnt_time_nights[night]
    fluxes = tnt_flux_nights[night]
    flux_err = tnt_flux_err_nights[night]

    plt.errorbar(times, fluxes, yerr=flux_err, fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.plot(times, model_function(initial_pos, times), 'g-', label="Initial Guess")
    plt.xlabel("$T-T_{0}$ (days)")
    plt.ylabel("Flux")
    plt.grid()
    plt.show()

    nwalkers, ndim = pos.shape
    if not Path(sampler_path / f'sampler_tnt_{night}.pkl').exists():
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=((times), (fluxes), (flux_err)))
        sampler.run_mcmc(pos, 10000, progress=True)

        # pickle the sampler
        with open(sampler_path / f'sampler_tnt_{night}.pkl', 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(sampler_path / f'sampler_tnt_{night}.pkl', 'rb') as f:
            sampler = pickle.load(f)

    samples = sampler.get_chain()

    labels = ["$m$","$c$", "$t\mathdefault{_0}$", "$t\mathdefault{_{i,0}}$", "$t\mathdefault{_{e,0}}$", "$a\mathdefault{_0}$", "$t\mathdefault{_1}$", "$t\mathdefault{_{i,1}}$", "$t\mathdefault{_{e,1}}$", "$a\mathdefault{_1}$", "$t\mathdefault{_2}$", "$t\mathdefault{_{i,2}}$", "$t\mathdefault{_{e,2}}$", "$a\mathdefault{_2}$", "$t\mathdefault{_3}$", "$t\mathdefault{_{i,3}}$", "$t\mathdefault{_{e,3}}$", "$a\mathdefault{_3}$", "$t\mathdefault{_4}$", "$t\mathdefault{_{i,4}}$", "$t\mathdefault{_{e,4}}$", "$a\mathdefault{_4}$"]
    labels = labels[:num_parameters]
    plot = False
    if plot:
        fig, axes = plt.subplots(num_parameters, figsize=(15, 2 * num_parameters), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])

        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        plt.show()

    # print(f"Acceptance Rate: {np.mean(sampler.acceptance_fraction)*100:.2f}%")
    # tau = sampler.get_autocorr_time()
    # print(tau)

    flat_samples = sampler.get_chain(discard=2000, thin=1, flat=True)

    if plot:
        plt.rcParams["font.size"] = 16
        fig = corner.corner(
            flat_samples, labels=labels
        )
        plt.show()
        plt.close()
        plt.rcParams["font.size"] = 32

    best_fits = []
    lower_uncertainties = []
    upper_uncertainties = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fits.append(mcmc[1])
        lower_uncertainties.append(q[0])
        upper_uncertainties.append(q[1])

    # print(f"m = ({best_fits[0]:.4f} - {lower_uncertainties[0]:.4f} + {upper_uncertainties[0]:.4f})")
    # print(f"c = ({best_fits[1]:.4f} - {lower_uncertainties[1]:.4f} + {upper_uncertainties[1]:.4f})")
    for i in range((num_parameters - 2) // 4):
        t_0s = flat_samples[:, 4 * i + 2]
        t_is = flat_samples[:, 4 * i + 3]
        t_es = flat_samples[:, 4 * i + 4]
        aes = flat_samples[:, 4 * i + 5]

        areas = integrate(aes, t_is, t_es)

        area = np.median(areas)
        area_lower, area_upper = np.diff(np.percentile(areas, [16, 50, 84]))

        t_mins = t_0s + t_es * t_is / (t_is + t_es) * np.log(t_es / t_is)
        t_min = np.median(t_mins)
        t_min_lower, t_min_upper = np.diff(np.percentile(t_mins, [16, 50, 84]))

        depths = 1 - (t_is + t_es) / (t_is + t_es + aes * t_is * (t_es / t_is) ** (t_es / (t_es + t_is)))
        depth = np.median(depths)
        depth_lower, depth_upper = np.diff(np.percentile(depths, [16, 50, 84]))

        equivalent_widths = areas / depth
        equivalent_width = np.median(equivalent_widths)
        equivalent_width_lower, equivalent_width_upper = np.diff(np.percentile(equivalent_widths, [16, 50, 84]))

        print(f"{t_min + ephemeris:.5f},{t_min_lower:.5f},{t_min_upper:.5f}", end=",")
        print(f"{depth:.5f},{depth_lower:.5f},{depth_upper:.5f}", end=",")
        print(f"{equivalent_width:5f},{equivalent_width_lower:.5f},{equivalent_width_upper:.5f}", end=",")
        print(f"{area:.5f},{area_lower:.5f},{area_upper:.5f}", end=",")
        print(f"{best_fits[4 * i + 2] + ephemeris:.5f},{lower_uncertainties[4 * i + 2]:.5f},{upper_uncertainties[4 * i + 2]:.5f}", end=",")
        print(f"{best_fits[4 * i + 3]:.5f},{lower_uncertainties[4 * i + 3]:.5f},{upper_uncertainties[4 * i + 3]:.5f}", end=",")
        print(f"{best_fits[4 * i + 4]:.5f},{lower_uncertainties[4 * i + 4]:.5f},{upper_uncertainties[4 * i + 4]:.5f}", end=",")
        print(f"{best_fits[4 * i + 5]:.5f},{lower_uncertainties[4 * i + 5]:.5f},{upper_uncertainties[4 * i + 5]:.5f},0,0,1")
    # print()
    # print([round(x, 4) for x in best_fits])

    plt.plot(times, model_function(best_fits, times), 'r-', label="Model")
    plt.errorbar(times, fluxes, yerr=flux_err, fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.xlabel("$T-T_{0}$ (days)")
    plt.ylabel("Flux")
    plt.grid()
    plt.show()

    plt.plot(times, model_function((0, 1, *best_fits[2:]), times), 'r-', label="Model")
    plt.errorbar(times, fluxes / (best_fits[0] * times + best_fits[1]), yerr=flux_err / (best_fits[0] * times + best_fits[1]), fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.xlabel("$T-T_{0}$ (days)")
    plt.ylabel("Flux")
    plt.grid()
    ylims = plt.gca().get_ylim()
    plt.show()

    plt.errorbar(times, 1 + fluxes / (best_fits[0] * times + best_fits[1]) - model_function((0, 1, *best_fits[2:]), times), yerr=flux_err / (best_fits[0] * times + best_fits[1]), fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.xlabel("$T-T_{0}$ (days)")
    plt.ylabel("Residual Flux")
    plt.ylim(ylims[0], ylims[1])
    plt.grid()
    plt.show()