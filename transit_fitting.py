from path import Path
import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten
import emcee
import corner
import pickle
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sampler_path = Path("/Users/nagro/PycharmProjects/w1m/mcmc/")

def model_function(theta, t):
    f_0 = theta[0]
    num_transits = (len(theta) - 1) // 4
    model = f_0 * np.ones_like(t)
    for i in range(num_transits):
        t_0 = theta[1 + i * 4]
        t_i = theta[2 + i * 4]
        t_e = theta[3 + i * 4]
        a = theta[4 + i * 4]
        model /= (a / (np.exp(-(t-t_0)/t_i) + np.exp((t-t_0)/t_e)) + 1)
    return model


def log_likelihood(theta, x, y, yerr):
    model = model_function(theta, x)
    return -0.5 * np.sum(((y - model) ** 2 / yerr ** 2) + np.log(2 * np.pi * yerr ** 2))


def log_prior(theta):
    f_0 = theta[0]
    t_0s = theta[1::4]
    t_is = theta[2:][::4]
    t_es = theta[3:][::4]
    amps = theta[4:][::4]
    if np.abs(f_0 - 1) > 0.1:
        return -np.inf
    if np.any(theta[1:] < 0):
        return -np.inf
    if np.any(t_is > 0.02):
        return -np.inf
    if np.any(t_es > 0.02):
        return -np.inf
    if np.any(amps > 0.5):
        return -np.inf
    return 0.0 # Flat prior


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

period = 14.804 / 24
ephemeris = 2460710.292987 - 4.92 / 24 + period / 2

# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_w1m.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["Time_BJD"] - ephemeris
w1m_flux = w1m_lc["Flux"]
w1m_flux_err = w1m_lc["Error"]

# tom's data
tom_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_tom.csv")
tom_lc = np.genfromtxt(tom_lc_path, delimiter=",", names=True)
tom_time = tom_lc["Time_BJD"] - ephemeris
tom_flux = tom_lc["Flux"]
tom_flux_err = tom_lc["Error"]

# tnt data
tnt_lc_path = Path("/Users/nagro/PycharmProjects/w1m/detrended_tnt.csv")
tnt_lc = np.genfromtxt(tnt_lc_path, delimiter=",", names=True)
tnt_time = tnt_lc["Time_BJD"] - ephemeris
tnt_flux = tnt_lc["Flux"]
tnt_flux_err = tnt_lc["Error"]

blue = "#648FFF"
orange = "#DC267F"
pink = "#FFB000"
colours = [blue, orange, pink]


initial_poses = [(1.003, -0.0277, 0.0004, 0.0014, 0.1166, -0.0021, 0.0029, 0.006, 0.3082),
                 (1.0074, 1.8445, 0.0005, 0.0129, 0.1577, 1.957, 0.0011, 0.0042, 0.2149, 1.9802, 0.0005, 0.0015, 0.1615, 2.0162, 0.0006, 0.0017, 0.1687, 2.0199, 0.0002, 0.0037, 0.087),
                 (1.0004, 3.0519, 0.0003, 0.002, 0.0756),
                 (1.045, 4.0217, 0.003, 0.0134, 0.1889),
                 (1.0106, 9.875, 0.0045, 0.0056, 0.2664, 9.9775, 0.001, 0.007, 0.1444),
                 (1.0055, 17.8914, 0.003, 0.0059, 0.262, 17.9973, 0.0007, 0.0027, 0.2654),
                 (1.0005, 26.7576, 0.0002, 0.0023, 0.1525),
                 (1.0051, 27.7662, 0.0049, 0.0033, 0.221, 27.8694, 0.0041, 0.0028, 0.1954, 27.992, 0.0011, 0.0019, 0.1505),
                 (1.0023, 28.9959, 0.0023, 0.0058, 0.2183),
                 (1,),
                 (1,),
                 (1.0135, 56.7513, 0.0005, 0.0083, 0.1794)
                 ]
initial_poses = [(1.0077, 20.3706, 0.0058, 0.0006, 0.2824, 20.469, 0.0012, 0.0032, 0.3294),
                 (1.0041, 21.3408, 0.0011, 0.0022, 0.2237),
                 (1.0021, 44.4185, 0.0009, 0.0083, 0.1759),
                 (1.0039, 46.2695, 0.0006, 0.0034, 0.1738),
                 (1.0114, 47.5029, 0.0012, 0.0113, 0.1278),
                 (1,),
                 (1.0026, 49.3553, 0.0016, 0.005, 0.1799),
                 ]
initial_poses = [(1.0011, 2.171, 0.0007, 0.0071, 0.206),
                 (1.0108, 3.1945, 0.0011, 0.0048, 0.2465, 3.2164, 0.0006, 0.0018, 0.1816),
                 (1.0039, 4.3215, 0.0034, 0.006, 0.3186),
                 (1.0059, 5.1686, 0.0003, 0.0059, 0.2209, 5.2642, 0.0007, 0.0067, 0.1671),
                 (1.0036, 10.1048, 0.0017, 0.0053, 0.1848, 10.2108, 0.0003, 0.0054, 0.171),
                 (1.0028, 13.1912, 0.0006, 0.0016, 0.2789, 13.3019, 0.0001, 0.0065, 0.1402),
                 (1.0002, 15.1586, 0.0012, 0.0035, 0.1341),
                 (1.0018, 16.1499, 0.0005, 0.004, 0.1807, 16.2751, 0.0003, 0.0047, 0.1609),
                 (1.0014, 17.2778, 0.0024, 0.0041, 0.42, 17.3841, 0.0014, 0.003, 0.2361),
                 (1.0016, 18.1261, 0.0005, 0.0035, 0.182, 18.2478, 0.0007, 0.0039, 0.1605),
                 (1.0024, 19.129, 0.0024, 0.0045, 0.3212, 19.2341, 0.0006, 0.0032, 0.27, 19.3605, 0.0007, 0.0035, 0.1445),
                 (0.9984, 20.3694, 0.0064, 0.0005, 0.1625),
                 (1.0011, 21.0835, 0.0009, 0.005, 0.2212, 21.2102, 0.0003, 0.0019, 0.1639),
                 (1.0019, 22.2187, 0.0055, 0.0029, 0.2463, 22.3196, 0.002, 0.0017, 0.3007),
                 (1.0005, 23.1949, 0.0017, 0.0011, 0.1731),
                 (1.013, 35.1674, 0.0026, 0.0097, 0.1699),
                 (1,),
                 (1,),
                 (1,),
                 (0.9994, 46.27, 0.0024, 0.0073, 0.186),
                 (1,),
                 (1.0008, 74.0217, 0.0009, 0.0042, 0.1631),
                 (0.9979, 75.254, 0.0006, 0.0031, 0.1052),
                 (1,),
                 (1,),
                 (1,),
                 (1.0, 80.1891, 0.0016, 0.0026, 0.1506),
                 (1,),
                 (1.0012, 82.0367, 0.0006, 0.0038, 0.1264),
                 (1,),
                 (1,)]

for night, initial_pos in enumerate(initial_poses):
    initial_pos = initial_poses[night]

    num_parameters = len(initial_pos)
    pos = initial_pos + 1e-5 * np.random.randn(48, num_parameters)

    # split the tnt flux into nights by looking for gaps of more than 0.5 days
    w1m_flux_err_nights = np.split(w1m_flux_err, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
    w1m_flux_nights = np.split(w1m_flux, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
    w1m_time_nights = np.split(w1m_time, np.where(np.diff(w1m_time) > 0.5)[0] + 1)

    times = w1m_time_nights[night]
    fluxes = w1m_flux_nights[night]
    flux_err = w1m_flux_err_nights[night]

    plt.errorbar(times, fluxes, yerr=flux_err, fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.plot(times, model_function(initial_pos, times), 'g-', label="Initial Guess")
    plt.xlabel("$T-T_{0}$ (days)")
    plt.ylabel("Flux")
    plt.grid()
    plt.show()
    if len(initial_pos) == 1:
        continue

    nwalkers, ndim = pos.shape
    if not Path(sampler_path / f'sampler_w1m_{night}.pkl').exists():
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=((times), (fluxes), (flux_err)))
        sampler.run_mcmc(pos, 10000, progress=True)

        # pickle the sampler
        with open(sampler_path / f'sampler_w1m_{night}.pkl', 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(sampler_path / f'sampler_w1m_{night}.pkl', 'rb') as f:
            sampler = pickle.load(f)

    samples = sampler.get_chain()

    labels = ["$f\mathdefault{_0}$", "$t\mathdefault{_0}$", "$t\mathdefault{_{i,0}}$", "$t\mathdefault{_{e,0}}$", "$a\mathdefault{_0}$", "$t\mathdefault{_1}$", "$t\mathdefault{_{i,1}}$", "$t\mathdefault{_{e,1}}$", "$a\mathdefault{_1}$", "$t\mathdefault{_2}$", "$t\mathdefault{_{i,2}}$", "$t\mathdefault{_{e,2}}$", "$a\mathdefault{_2}$", "$t\mathdefault{_3}$", "$t\mathdefault{_{i,3}}$", "$t\mathdefault{_{e,3}}$", "$a\mathdefault{_3}$", "$t\mathdefault{_4}$", "$t\mathdefault{_{i,4}}$", "$t\mathdefault{_{e,4}}$", "$a\mathdefault{_4}$"]
    # labels = labels[:num_parameters]
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

    plt.rcParams["font.size"] = 16
    fig = corner.corner(
        flat_samples, labels=labels
    )
    plt.show()
    plt.close()
    plt.rcParams["font.size"] = 32

    best_fits = []
    uncertainties = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fits.append(mcmc[1])
        uncertainties.append((q[0] + q[1]) / 2)

    print(f"f0 = ({best_fits[0]:.4f} ± {uncertainties[0]:.4f})")
    for i in range((num_parameters - 1) // 4):
        print(f"t0 = ({best_fits[4 * i + 1]:.4f} ± {uncertainties[4 * i + 1]:.4f})")
        print(f"ti = ({best_fits[4 * i + 2]:.4f} ± {uncertainties[4 * i + 2]:.4f})")
        print(f"te = ({best_fits[4 * i + 3]:.4f} ± {uncertainties[4 * i + 3]:.4f})")
        print(f"a = ({best_fits[4 * i + 4]:.4f} ± {uncertainties[4 * i + 4]:.4f})")
    print()
    print([round(x, 4) for x in best_fits])

    plt.plot(times, model_function([1, *best_fits[1:]], times), 'r-', label="Model")
    # plt.plot(times, model_function(initial_pos, times), 'g-', label="Uncertainty")
    plt.errorbar(times, fluxes / best_fits[0], yerr=flux_err, fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.xlabel("T - T$_{0}$ (days)")
    plt.ylabel("Flux")
    plt.show()

    plt.errorbar(times, fluxes - model_function(best_fits, times), yerr=flux_err, fmt='o', color=colours[1], label="Data", alpha=0.05)
    plt.xlabel("T - T$_{0}$ (days)")
    plt.ylabel("Residuals")
    plt.axhline(0, color='k', linestyle='--')
    plt.show()
    exit()