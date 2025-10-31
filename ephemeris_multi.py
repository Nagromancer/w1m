# script to generate a plot of the evolution of the transit parameters
from path import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

source_path = Path("/Users/nagro/PycharmProjects/w1m/scratch.csv")
plot_path = Path("/Users/nagro/PycharmProjects/w1m/ephemeris_plots")
source = np.genfromtxt(source_path, delimiter=",", names=True)

transit_id = 7
source = source[source["id"] == transit_id]

# plot depth against t_min
plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

t0_errors = np.array(list(zip(source["t_c_minus"], source["t_c_plus"]))).T
area_errors = np.array(list(zip(source["area_minus"], source["area_plus"]))).T


# plot area vs t0
plt.figure()
plt.errorbar((source["t_c"] - 2460000), source["area"] * 86400, xerr=t0_errors, yerr=area_errors * 86400, fmt="o", color="black", markersize=5, capsize=5)
plt.xlabel("$t_0$ (BJD - 2460000)")
plt.ylabel("Equivalent area (seconds)")
plt.title(f"Area vs $t_0$ for transit {transit_id}")
plt.tight_layout()
plt.ylim(0, 260)
plt.grid()
plt.show()

def fit_transit_period_weighted(transit_times, uncertainties, P0, T0=None, plot=False):
    """
    Fit orbital period using weighted least squares and return uncertainties.

    Returns:
        P_fit (float), P_err (float): Best-fit orbital period and uncertainty.
        T0_fit (float), T0_err (float): Best-fit reference time and uncertainty.
        residuals (np.array): O - C residuals.
    """
    transit_times = np.array(transit_times)
    uncertainties = np.array(uncertainties)

    if T0 is None:
        T0 = transit_times[0]

    n = np.round((transit_times - T0) / P0).astype(int)
    A = np.vstack([n, np.ones_like(n)]).T
    w = 1.0 / uncertainties ** 2
    W = np.diag(w)

    # Weighted least squares solution
    AtW = A.T @ W
    cov = np.linalg.inv(AtW @ A)
    coeffs = cov @ (AtW @ transit_times)
    P_fit, T0_fit = coeffs

    # Parameter uncertainties from covariance matrix
    P_err = np.sqrt(cov[0, 0])
    T0_err = np.sqrt(cov[1, 1])

    # Residuals
    T_model = T0_fit + n * P_fit
    residuals = transit_times - T_model

    if plot:
        plt.errorbar(n, residuals * 24 * 60, yerr=uncertainties * 24 * 60,
                     fmt='o', capsize=3, color='navy')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [minutes]")
        plt.title(f"O-C Diagram (Transit {transit_id})")
        plt.text(0.35, 0.15, rf"$P={P_fit*24:.6f}\pm{P_err*24:.6f}$"" hours", transform=plt.gca().transAxes, fontsize=32)
        plt.text(0.35, 0.05, rf"$T_0={T0_fit:.5f}\pm{T0_err:.5f}$"" BJD", transform=plt.gca().transAxes, fontsize=32)
        plt.grid(True)
        # plt.ylim(-25, 25)
        plt.savefig(plot_path / f"transit_{transit_id}_linear.png", dpi=200)
        plt.show()

    print(f"Best-fit period: {P_fit:.7f} ± {P_err:.7f} days")
    print(f"Best-fit T0:     {T0_fit:.5f} ± {T0_err:.5f} BJD")
    return P_fit, P_err, T0_fit, T0_err, residuals


def fit_period_decaying(transit_times, uncertainties, P0, T0=None, plot=False):
    """
    Fit orbital period using decaying period and return uncertainties.

    Returns:
        P_fit (float), P_err (float): Best-fit orbital period and uncertainty.
        T0_fit (float), T0_err (float): Best-fit reference time and uncertainty.
        residuals (np.array): O - C residuals.
    """
    transit_times = np.array(transit_times)
    uncertainties = np.array(uncertainties)

    if T0 is None:
        T0 = transit_times[0]

    n = np.round((transit_times - T0) / P0).astype(int)

    def transit_model(n, T0, P, Pdot):
        return T0 + n * P + 0.5 * Pdot * n**2

    # Fit the model
    popt, pcov = curve_fit(transit_model, n, transit_times, sigma=uncertainties, absolute_sigma=True, p0=[T0, P0, 0.0])

    # Extract parameters and uncertainties
    T0_fit, P_fit, Pdot_fit = popt
    T0_err, P_err, Pdot_err = np.sqrt(np.diag(pcov))
    p_dot_over_p_unc = np.abs(Pdot_fit / P_fit * np.sqrt((Pdot_err / Pdot_fit)**2 + (P_err / P_fit)**2))

    dp_dt = 86400 * Pdot_fit / P_fit  # seconds per day
    dp_dt_e = 86400 * p_dot_over_p_unc  # seconds per day

    print(f"T0 = {T0_fit:.5f} ± {T0_err:.5f} BJD")
    print(f"P = {P_fit:.6f} ± {P_err:.6f} day")
    # print(f"Pdot = {Pdot_fit:.8f} ± {Pdot_err:.8f} day/transit^2")
    print(f"dP/dt = {dp_dt:.3f} ± {dp_dt_e:.3f} seconds per day")

    if plot:
        n_fit = np.linspace(n.min(), n.max(), 100)
        plt.errorbar(n, (transit_times - transit_model(n, T0_fit, P_fit, Pdot_fit)) * 1440, yerr=uncertainties * 1440,
                     fmt='o', capsize=3, color='navy')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [min]")
        plt.title(f"O-C Diagram with Decaying Period (Transit {transit_id})")
        plt.text(0.5, 0.05, rf"$\frac{{dP}}{{dt}}={dp_dt:.3f}\pm{dp_dt_e:.3f}$"" s day$^{-1}$", transform=plt.gca().transAxes, fontsize=32)
        # plt.ylim(-25, 25)
        plt.grid()
        plt.savefig(plot_path / f"transit_{transit_id}_decaying.png", dpi=200)
        plt.show()


transits = source["t_c"]
uncerts = (source["t_c_minus"] + source["t_c_plus"]) / 2
P0 = 14.8029 / 24  # initial period guess in days

P_fit, P_err, T0_fit, T0_err, residuals = fit_transit_period_weighted(transits, uncerts, P0, plot=True)
print(f"Fitting decaying period model:")
try:
    fit_period_decaying(transits, uncerts, P0, T0=T0_fit, plot=True)
except:
    pass

# P_fit, P_err, T0_fit, T0_err, residuals = fit_transit_period_weighted(transits[-9:], uncerts[-9:], P0, plot=True)
