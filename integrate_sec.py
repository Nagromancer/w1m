import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def model_function(t, theta):
    product = np.ones_like(t)
    for i in range(len(theta) // 4):
        t_0, t_i, t_e, a = theta[4*i:4*(i+1)]
        product *= 1 / (a / (np.exp(-(t-t_0) / t_i) + np.exp((t-t_0) / t_e)) + 1)
    return 1 - product


def integrate(theta, n=1000):
    # every fourth parameter corresponds to a different section
    t_0s = np.array(theta[0::4])
    t_is = np.array(theta[1::4])
    t_es = np.array(theta[2::4])
    lower_bound = np.min(t_0s - 20 * t_is)
    upper_bound = np.max(t_0s + 20 * t_es)

    t = np.linspace(lower_bound, upper_bound, n)
    flux = model_function(t, theta)
    flux_t = t * flux
    integral = np.trapezoid(flux, t)
    flux_t_integral = np.trapezoid(flux_t, t)
    centroid = flux_t_integral / integral

    return integral, centroid
