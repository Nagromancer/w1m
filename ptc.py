import glob
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import gridspec
from path import Path
from scipy.optimize import curve_fit

import sys

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'Times'
plt.rc('text', usetex=True)

def load_image_data(path):
    """
    Load image data from FITS files in the specified path.
    """
    list_of_signals = glob.glob(os.path.join(path, 'ptc*.fits'))
    list_of_signals.sort()
    paired_signals = [list_of_signals[i:i + 2] for i in range(0, len(list_of_signals), 2)]
    master_bias = fits.getdata(os.path.join(path, 'master_bias.fits'))
    image_1, image_2 = [], []
    exposures = []
    region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                       [x.split(':') for x in (fits.getheader(list_of_signals[0])['IMAG-RGN'].split(','))]])
    region[:, 1] += 1
    for pair in paired_signals:
        data_1, header_1 = fits.getdata(pair[0], header=True)
        data_2, header_2 = fits.getdata(pair[1], header=True)
        data_1 = data_1[region[1][0]:region[1][1], region[0][0]:region[0][1]].astype(np.float32)
        data_2 = data_2[region[1][0]:region[1][1], region[0][0]:region[0][1]].astype(np.float32)

        # Subtract master bias
        data_1 = (data_1 - master_bias)[1000:1200, 1000:1200]
        data_2 = (data_2 - master_bias)[1000:1200, 1000:1200]
        # Ensure data is 2D
        if data_1.ndim != 2 or data_2.ndim != 2:
            raise ValueError(f"File {pair[0]} or {pair[1]} is not 2D.")

        image_1.append(data_1)
        image_2.append(data_2)
        exposure_1 = header_1.get('EXPTIME', 'N/A')
        exposure_2 = header_2.get('EXPTIME', 'N/A')
        exposure = (exposure_1 + exposure_2) / 2
        exposures.append(exposure)
        logging.log(logging.INFO,
            f"Pair: {os.path.basename(pair[0])}, {os.path.basename(pair[1])} -> Exposures: {exposure_1}, {exposure_2}")

    logging.log(logging.INFO, f"Loaded {len(image_1)} pairs of images. Each image shape: {image_1[0].shape}")
    return np.array(image_1), np.array(image_2), np.array(exposures)


def calculate_average(image_1, image_2):
    """
    Calculate the average pixel intensity for cropped regions in two image sets.
    """
    image_1_mean = [np.mean(img) for img in image_1]
    image_2_mean = [np.mean(img) for img in image_2]
    average = np.array([(i + j) / 2 for i, j in zip(image_1_mean, image_2_mean)])
    return average


def calculate_variance(image_1, image_2):
    """
    Calculate variance for two image sets.
    """
    variance = [((i - j) - (k - l)) ** 2 / 2 for i, j, k, l in
                zip(image_1, np.mean(image_1, axis=(1, 2)), image_2, np.mean(image_2, axis=(1, 2)))]
    variance = np.array([np.mean(var) for var in variance])
    return variance


def ptc_fit_high(x, a_1, b_1):
    """
    Linear fit function for PTC curve.
    """
    return a_1 * x + b_1


def fit_ptc_curve(average, variance, max_linear=55000):
    """
    Fit the PTC curve using a linear model, ignoring nonlinear regions.

    Parameters
    ----------
    average : numpy.ndarray
        Average pixel values.
    variance : numpy.ndarray
        Variance of pixel values.
    max_linear : float
        Maximum value to consider for linear fit (to exclude saturation effects).

    Returns
    -------
    tuple
        Fitted parameters and gain.
    """
    # Reject nonlinear regions
    mask = average < max_linear
    x_1 = average[mask]
    y_1 = variance[mask]

    # Perform the linear fit
    popt_1, pcov_1 = curve_fit(ptc_fit_high, x_1, y_1)
    gain = 1 / popt_1[0]
    gain_uncertainty = 1 / (-np.sqrt(pcov_1[0][0])+popt_1[0]) - gain

    logging.log(logging.INFO, f"Linear fit to ptc: slope = {popt_1[0]:.6f}, intercept = {popt_1[1]:.6f}")
    logging.log(logging.INFO, f"Gain: {gain:.3f} ± {gain_uncertainty:.3f} e-/ADU")
    return popt_1, gain


def plot_ptc_curve(average, variance, gain, popt_1, max_linear=55000):
    """
    Plot the PTC curve.
    """
    # Identify the saturation point
    max_index = np.argmax(variance)
    saturation_grey_value = average[max_index]
    variance_sqr_saturation_grey = variance[max_index]
    logging.log(logging.INFO, f'Well Depth = {saturation_grey_value * gain:4.0f} e-')

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.plot(average, variance, 'ko')
    mask = (average < max_linear)
    x_fit = average[mask]
    ax1.plot(x_fit, ptc_fit_high(x_fit, *popt_1), 'r-', label=rf'Linear fit')
    ax1.axvline(x=x_fit[-1], color='r', linestyle='--', linewidth=1)
    ax1.axvline(x=x_fit[0], color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('')
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_ylabel('Variance (ADU$^\mathdefault{2}$)')
    ax1.legend(loc='upper left')
    ax1.axvline(x=saturation_grey_value, color='b', linestyle='--', linewidth=1)
    ax2.set_xlabel('Pixel count (ADU)')
    ax2.set_ylabel('Residuals (ADU$^\mathdefault{2}$)')
    ax2.plot(average, variance - ptc_fit_high(average, *popt_1), 'ko')
    ax2.axvline(x=saturation_grey_value, color='b', linestyle='--', linewidth=1)
    ax2.axvline(x=x_fit[-1], color='r', linestyle='--', linewidth=1)
    ax2.axvline(x=x_fit[0], color='r', linestyle='--', linewidth=1)
    ax2.set_ylim(-450, 450)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)

    fig.tight_layout()
    fig.savefig(path / f'{out_base}-ptc.pdf', bbox_inches='tight')
    plt.show()


def calculate_gain_standard_deviation(n_pixels):
    """
    Calculate the standard deviation of the gain.
    """
    sigma_gain = 100 * np.sqrt(8 / n_pixels)
    return sigma_gain


def calculate_ptc(path, max_linear):
    """
    Main function to calculate and save the PTC.
    """
    image_1, image_2, exposures = load_image_data(path)
    average = calculate_average(image_1, image_2)
    variance = calculate_variance(image_1, image_2)
    popt_1, gain = fit_ptc_curve(average, variance, max_linear=max_linear)

    max_index = np.argmax(variance)
    x_max = average[max_index]
    saturation_value = x_max

    plot_ptc_curve(average, variance, gain, popt_1, max_linear=max_linear)

    return saturation_value, average, variance, exposures


def plot_linearity_line(gradient, offset, startx, endx, step, figure, ax1):
    plt.figure(figure)
    x_values = []
    y_values = []

    for x in np.arange(startx, endx, step):
        y = x * gradient + offset  # y = mx + c
        x_values.append(x)
        y_values.append(y)
    ax1.plot(x_values, y_values, 'r-', label=f'Linear fit')


def plot_linearity(exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, LinearityError, ResidualsList, corrected_counts, figure):
    startx = (min(ExposureTimeList_5_95))
    endx = (max(ExposureTimeList_5_95))
    step = 0.0001

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], wspace=0, hspace=0)
    figure = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_ylabel('Signal (ADU)')
    ax1.plot([exposure_times], [corrected_counts], 'ko')
    plot_linearity_line(Linearitygradient, LinearityOffset, startx, endx, step, figure, ax1)
    ax1.axvline(x=startx, color='r', linestyle='--', linewidth=1)
    ax1.axvline(x=endx, color='r', linestyle='--', linewidth=1)
    ax1.legend(loc='lower right')

    ax2.plot(exposure_times, ResidualsList, 'ko', linewidth=1)
    ax2.plot([startx, endx], [0, 0], 'r-', linewidth=1)
    ax2.set_ylim(-3 * LinearityError, + 3 * LinearityError)
    ax2.set_ylabel('Residuals (\%)')
    ax2.set_xlabel('Exposure (s)')
    ax2.axvline(x=startx, color='r', linestyle='--', linewidth=1)
    ax2.axvline(x=endx, color='r', linestyle='--', linewidth=1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    figure.savefig(path / f'{out_base}-linearity.pdf', bbox_inches='tight')
    plt.show()


def set_best_fit_ranges(xdata, ydata, startx, endx):
    best_fit_xdata = []
    best_fit_ydata = []

    for x, y in zip(xdata, ydata):
        if startx < x < endx:
            best_fit_xdata.append(x)
            best_fit_ydata.append(y)

    return best_fit_xdata, best_fit_ydata


def best_fit(xdata, ydata):
    def func(x, a, b):
        return a * x + b

    Gradient = curve_fit(func, xdata, ydata)[0][0]
    Offset = curve_fit(func, xdata, ydata)[0][1]

    logging.log(logging.INFO, f"Linear fit to linearity: slope = {Gradient:.6f}, intercept = {Offset:.6f}")
    return Gradient, Offset


class CreateResiduals():
    def __init__(self, X_values, Y_values, Offset, Gradient, max_val, range_factor):
        self.x = X_values
        self.y = Y_values
        self.offset = Offset
        self.gradient = Gradient
        self.max = max_val
        self.range = range_factor
        self.residuals = []
        self.residual_counts = []

        self.create()

    def create(self):
        for i in range(0, len(self.y)):
            calculated_level = self.offset + (self.gradient * self.x[i])

            Residuals = (self.y[i] - calculated_level) / (
                    self.range * self.max) * 100  # Equation 35 from EMVA1288-V3.1

            residualscts = (self.y[i] - calculated_level)

            self.residuals.append(Residuals)
            self.residual_counts.append(residualscts)


def find_linearity_error(ResidualsList):
    LinearityError = (max(ResidualsList) - min(ResidualsList)) / 2

    return LinearityError


warnings.filterwarnings('ignore')

# Set paths
path_str = '/Volumes/SanDisk-2TB-SSD/w1m/ptc/ptc_blue_4'
out_base = path_str.split('/')[-1]
path = Path(path_str)

# start logging to file
logging.basicConfig(
    filename=path / 'ptc.log',
    level=logging.INFO,
    format='%(message)s',  # Custom format to remove INFO:root:
    filemode='w'  # Clear the log file
)

# Create a StreamHandler to print to the terminal with the same format
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Add the handler to the root logger
logging.getLogger().addHandler(console_handler)

# Calculate PTC
ptc_limit = 55000
saturation_value, average, variance, exposures = calculate_ptc(path, max_linear=ptc_limit)

figure = 2
startx = saturation_value * 0.05
endx = saturation_value * 0.95

CorrectedCtsList_5_95, ExposureTimeList_5_95 = set_best_fit_ranges(average, exposures, startx, endx)
Linearitygradient, LinearityOffset = best_fit(ExposureTimeList_5_95, CorrectedCtsList_5_95)

range_factor = 0.9

ResidualsList = CreateResiduals(exposures, average, LinearityOffset, Linearitygradient,
                                saturation_value, range_factor).residuals
ResidualsList_5_95 = CreateResiduals(ExposureTimeList_5_95, CorrectedCtsList_5_95, LinearityOffset, Linearitygradient,
                                     saturation_value, range_factor).residuals

LinearityError = find_linearity_error(ResidualsList_5_95)
logging.log(logging.INFO, f'Linearity Error: {LinearityError:.3f}%')

plot_linearity(exposures, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, LinearityError, ResidualsList, average, figure)