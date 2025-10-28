import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.dpi'] = 200


def model_function(t, t0, t_i, t_e, a):
    return 1 - 1 / (a / (np.exp(-(t-t0) / t_i) + np.exp((t-t0) / t_e)) + 1)


def fast_integrate(t0, t_i, t_e, a, n=400):
    t = np.linspace(-20 * t_i, 20 * t_e, n)
    flux = model_function(t, 0, t_i, t_e, a)
    flux_t = t * flux
    integral = np.trapezoid(flux, t)
    flux_t_integral = np.trapezoid(flux_t, t)
    centroid = flux_t_integral / integral + t0
    return integral, centroid


def integrate(theta):
    integrals = np.zeros(len(theta) // 4)
    centroids = np.zeros(len(theta) // 4)
    for i in range(len(theta) // 4):
        t_0, t_i, t_e, a = theta[4*i:4*(i+1)]
        integrals[i], centroids[i] = fast_integrate(t_0, t_i, t_e, a)
    total_area = integrals.sum()
    total_centroid = np.sum(centroids * integrals) / total_area
    return total_area, total_centroid
