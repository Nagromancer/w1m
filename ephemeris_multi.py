# script to generate a plot of the evolution of the transit parameters
from path import Path
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

source_path = Path("/Users/nagro/PycharmProjects/w1m/transit_params_multi.csv")
plot_path = Path("/Users/nagro/PycharmProjects/w1m/ephemeris_plots")
source = np.genfromtxt(source_path, delimiter=",", names=True)

np.random.seed(42)

# plot depth against t_min
plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200


transit_id = 3
source = source[source["id"] == transit_id]
t0_errors = np.array(list(zip(source["t_c_minus"], source["t_c_plus"]))).T
area_errors = np.array(list(zip(source["area_minus"], source["area_plus"]))).T

area_plot = False
if area_plot:
    # plot area vs t0
    plt.figure()
    plt.errorbar((source["t_c"] - 2460000), source["area"] * 86400, xerr=t0_errors, yerr=area_errors * 86400, fmt="o", color="black", markersize=5, capsize=5)
    plt.xlabel("$t_0$ (BJD - 2460000)")
    plt.ylabel("Equivalent area (seconds)")
    plt.title(f"Area vs $t_0$ for transit {transit_id}")
    plt.tight_layout()
    plt.ylim(0, 260)
    plt.grid()
    plt.tight_layout()
    plt.show()


def fit_period_decaying(transit_times, uncertainties, initial, plot=False):
    """
    Fit orbital period using decaying period and return uncertainties.

    Returns:
        P_fit (float), P_err (float): Best-fit orbital period and uncertainty.
        T0_fit (float), T0_err (float): Best-fit reference time and uncertainty.
        residuals (np.array): O - C residuals.
    """
    def decaying_transit_model(theta, n):
        T0, P, Pdot = theta
        return T0 + n * P + 0.5 * Pdot * n ** 2

    def log_likelihood(theta, n, t, terr):
        ln_s = theta[-1]
        s2 = np.exp(2 * ln_s)
        model = decaying_transit_model(theta[:-1], n)
        var = terr ** 2 + s2
        residuals = t - model
        return -0.5 * np.sum(residuals ** 2 / var + np.log(2 * np.pi * var))

    def log_prior(theta):
        T0, P, Pdot, ln_s = theta
        if -10 < ln_s < 1:
            return 0.0
        return -np.inf

    def log_probability(theta, n, t, terr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, n, t, terr)

    labels = ["$T\mathdefault{_0}$", "$P\mathdefault{_0}$", "$\dot P$", "$\ln\,s$"]
    ndim = len(labels)

    transit_times = np.array(transit_times)
    uncertainties = np.array(uncertainties)

    T0, P, Pdot, ln_s = initial
    n = np.round((transit_times - T0) / P0).astype(int)

    nwalkers = 50
    nsteps = 1e4
    burnin = 2e3
    p0 = np.array([T0, P, Pdot, ln_s]) + 1e-4 * np.random.randn(nwalkers, 4)

    # Set up and run MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(n, transit_times, uncertainties))
    sampler.run_mcmc(p0, int(nsteps))
    samples = sampler.get_chain(discard=int(burnin))
    flat_samples = sampler.get_chain(discard=int(burnin), flat=True)

    best_fits = []
    lower_uncertainties = []
    upper_uncertainties = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fits.append(mcmc[1])
        lower_uncertainties.append(q[0])
        upper_uncertainties.append(q[1])

    fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step Number")
    plt.tight_layout()
    plt.show()

    plt.rcParams["font.size"] = 16
    fig = corner.corner(
        flat_samples, labels=labels
    )
    fig.suptitle(f"Posterior Distributions", fontsize=20)
    plt.show()
    plt.close()
    plt.rcParams["font.size"] = 32

    for i in range(ndim):
        print(f"{labels[i]} = {best_fits[i]:.8f} +{upper_uncertainties[i]:.8f} -{lower_uncertainties[i]:.8f}")

    # separate the flat samples for each parameter for further analysis
    T0_samples = flat_samples[:, 0]
    P_samples = flat_samples[:, 1]
    Pdot_samples = flat_samples[:, 2]
    ln_s_samples = flat_samples[:, 3]

    dp_dt_samples = 86400 * Pdot_samples / P_samples  # seconds per day
    mcmc_dp_dt = np.percentile(dp_dt_samples, [16, 50, 84])
    dp_dt = mcmc_dp_dt[1]
    dp_dt_lower, dp_dt_upper = np.diff(mcmc_dp_dt)

    print(f"dP/dt = {dp_dt:.3f} +{dp_dt_upper:.3f} -{dp_dt_lower:.3f} s day^-1")

    # convert from ln s to s in seconds
    print(f"s = {np.exp(best_fits[-1]) * 86400:.0f} +{(np.exp(best_fits[-1] + upper_uncertainties[-1]) - np.exp(best_fits[-1])) * 86400:.0f} -{(np.exp(best_fits[-1]) - np.exp(best_fits[-1] - lower_uncertainties[-1])) * 86400:.0f} seconds")

    s2 = np.exp(2 * best_fits[-1])
    uncertainties = np.sqrt(uncertainties**2 + s2)

    if plot:
        plt.errorbar(n, (transit_times - decaying_transit_model(best_fits[:-1], n)) * 86400, yerr=uncertainties * 86400,
                     fmt='o', capsize=3, color='navy')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [seconds]")
        plt.title(f"O-C Diagram with Decaying Period (Transit {transit_id})")
        # plt.text(0.5, 0.05, rf"$\frac{{dP}}{{dt}}={dp_dt:.3f}\pm{dp_dt_e:.3f}$"" s day$^{-1}$", transform=plt.gca().transAxes, fontsize=32)
        # plt.ylim(-25, 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_path / f"transit_{transit_id}_decaying.png", dpi=200)
        plt.show()


def fit_period_linear(transit_times, uncertainties, initial, plot=False):
    """
    Fit orbital period using linear period and return uncertainties.

    Returns:
        P_fit (float), P_err (float): Best-fit orbital period and uncertainty.
        T0_fit (float), T0_err (float): Best-fit reference time and uncertainty.
        residuals (np.array): O - C residuals.
    """
    def linear_transit_model(theta, n):
        T0, P = theta
        return T0 + n * P

    def log_likelihood(theta, n, t, terr):
        ln_s = theta[-1]
        s2 = np.exp(2 * ln_s)
        model = linear_transit_model(theta[:-1], n)
        var = terr ** 2 + s2
        residuals = t - model
        return -0.5 * np.sum(residuals ** 2 / var + np.log(2 * np.pi * var))

    def log_prior(theta):
        T0, P, ln_s = theta
        if -10 < ln_s < 1:
            return 0.0
        return -np.inf

    def log_probability(theta, n, t, terr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, n, t, terr)

    labels = ["$T\mathdefault{_0}$", "$P\mathdefault{_0}$", "$\ln\,s$"]
    ndim = len(labels)

    transit_times = np.array(transit_times)
    uncertainties = np.array(uncertainties)

    T0, P, ln_s = initial
    n = np.round((transit_times - T0) / P0).astype(int)

    nwalkers = 50
    nsteps = 2e4
    burnin = 1e4
    p0 = np.array([T0, P, ln_s]) + 1e-4 * np.random.randn(nwalkers, ndim)

    # Set up and run MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(n, transit_times, uncertainties))
    sampler.run_mcmc(p0, int(nsteps))
    samples = sampler.get_chain(discard=int(burnin))
    flat_samples = sampler.get_chain(discard=int(burnin), flat=True)

    best_fits = []
    lower_uncertainties = []
    upper_uncertainties = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fits.append(mcmc[1])
        lower_uncertainties.append(q[0])
        upper_uncertainties.append(q[1])

    fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step Number")
    plt.tight_layout()
    plt.show()

    plt.rcParams["font.size"] = 16
    fig = corner.corner(
        flat_samples, labels=labels
    )
    fig.suptitle(f"Posterior Distributions", fontsize=20)
    plt.show()
    plt.close()
    plt.rcParams["font.size"] = 32

    for i in range(ndim):
        print(f"{labels[i]} = {best_fits[i]:.6f} +{upper_uncertainties[i]:.6f} -{lower_uncertainties[i]:.6f}")


    # convert from ln s to s in seconds
    print(f"s = {np.exp(best_fits[-1]) * 86400:.0f} +{(np.exp(best_fits[-1] + upper_uncertainties[-1]) - np.exp(best_fits[-1])) * 86400:.0f} -{(np.exp(best_fits[-1]) - np.exp(best_fits[-1] - lower_uncertainties[-1])) * 86400:.0f} seconds")

    s2 = np.exp(2 * best_fits[-1])
    uncertainties = np.sqrt(uncertainties**2 + s2)

    if plot:
        plt.errorbar(n, (transit_times - linear_transit_model(best_fits[:-1], n)) * 86400, yerr=uncertainties * 86400,
                     fmt='o', capsize=3, color='navy')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [seconds]")
        plt.title(f"O-C Diagram with Linear Period (Transit {transit_id})")
        # plt.text(0.5, 0.05, rf"$\frac{{dP}}{{dt}}={dp_dt:.3f}\pm{dp_dt_e:.3f}$"" s day$^{-1}$", transform=plt.gca().transAxes, fontsize=32)
        # plt.ylim(-25, 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_path / f"transit_{transit_id}_linear.png", dpi=200)
        plt.show()


transits = source["t_c"]
uncerts = (source["t_c_minus"] + source["t_c_plus"]) / 2
P0 = 14.8029 / 24  # initial period guess in days
T0 = source["t_c"][0]  # initial T0 guess
initial = (T0, P0, 0, -5)
print(f"Performing linear fit for transit {transit_id}...")
fit_period_linear(transits, uncerts, (T0, P0, -5), plot=True)

print(f"Performing decaying fit for transit {transit_id}...")
fit_period_decaying(transits, uncerts, initial, plot=True)
