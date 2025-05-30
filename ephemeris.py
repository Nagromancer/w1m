# script to generate a plot of the evolution of the transit parameters
from path import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

source_path = Path("/Users/nagro/PycharmProjects/w1m/transit_params.csv")
source = np.genfromtxt(source_path, delimiter=",", names=True)

transit_id = 2
source = source[source["id"] == transit_id]

# plot depth against t_min
plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200

t_min_errors = np.array(list(zip(source["t_min_minus"], source["t_min_plus"]))).T
depth_errors = np.array(list(zip(source["depth_minus"], source["depth_plus"]))).T
t0_errors = np.array(list(zip(source["t0_minus"], source["t0_plus"]))).T
area_errors = np.array(list(zip(source["area_minus"], source["area_plus"]))).T

# plot o - c against t_min
period = 14.8029 / 24
plt.figure()

observed = source["t_min"] - 2460710.39967
o_minus_c = ((observed + period / 2) % period - period / 2) * 1440
plt.errorbar(source["t_min"] - 2.46e6, o_minus_c, xerr=t_min_errors, yerr=t_min_errors * 1440, fmt="o", color="black", markersize=5, capsize=5)
plt.xlabel("Max Depth Time (BJD - 2460000)")
plt.ylabel("O - C (min)")
plt.grid()
plt.show()

# plot area vs t0
plt.figure()
plt.errorbar(source["t0"] - 2460000, source["area"], xerr=t0_errors, yerr=area_errors, fmt="o", color="black", markersize=5, capsize=5)
plt.xlabel("$t_0$ (BJD - 2460000)")
plt.ylabel("Equivalent area (days)")
plt.title(f"Equivalent width vs $t_0$ for transit {transit_id}")
plt.tight_layout()
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
        plt.title("O-C Diagram (Weighted Fit)")
        plt.grid(True)
        plt.show()

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

    if plot:
        n_fit = np.linspace(n.min(), n.max(), 100)
        T_model = transit_model(n_fit, *popt)
        plt.errorbar(n, (transit_times - transit_model(n, *popt)) * 1440, yerr=uncertainties * 1440,
                     fmt='o', capsize=3, color='navy')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [days]")
        plt.show()

    print(f"T0 = {T0_fit:.6f} ± {T0_err:.6f}")
    print(f"P = {P_fit:.6f} ± {P_err:.6f}")
    print(f"Pdot / P = {86400 * Pdot_fit / P_fit:.6f} ± {86400 * np.sqrt(Pdot_err**2 + P_err**2):.6f} seconds per day")


transits = source["t0"]
uncerts = (source["t0_minus"] + source["t0_plus"]) / 2
P0 = 14.8029 / 24  # initial period guess in days


P_fit, P_err, T0_fit, T0_err, residuals = fit_transit_period_weighted(transits, uncerts, P0, plot=True)
fit_period_decaying(transits, uncerts, P0, T0=T0_fit, plot=True)

P_fit, P_err, T0_fit, T0_err, residuals = fit_transit_period_weighted(transits[-8:], uncerts[-8:], P0, plot=True)
print(f"Best-fit period: {P_fit:.6f} ± {P_err:.6f} days")
print(f"Best-fit T0:     {T0_fit:.5f} ± {T0_err:.5f} BJD")
