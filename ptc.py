#!/usr/bin/env python
import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import configparser
from matplotlib import gridspec

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'Times'
plt.rc('text', usetex=True)

warnings.filterwarnings('ignore')

def load_image_data(path):
    """
    Load image data from FITS files in the specified path.
    """
    list_of_signals = glob.glob(os.path.join(path, 'ptc*.fits'))
    list_of_signals.sort()
    paired_signals = [list_of_signals[i:i + 2] for i in range(0, len(list_of_signals), 2)]
    # paired_signals = [pair for idx, pair in enumerate(paired_signals) if idx not in [0]]
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
        print(
            f"  Pair: {os.path.basename(pair[0])}, {os.path.basename(pair[1])} -> Exposures: {exposure_1}, {exposure_2}")

    print(f"Loaded {len(image_1)} pairs of images. Each image shape: {image_1[0].shape}")
    return np.array(image_1), np.array(image_2), np.array(exposures)


def calculate_average(image_1, image_2):
    """
    Calculate the average pixel intensity for cropped regions in two image sets.
    """
    image_1_mean = [np.mean(img) for img in image_1]
    image_2_mean = [np.mean(img) for img in image_2]
    average = np.array([(i + j) / 2 for i, j in zip(image_1_mean, image_2_mean)])
    print('Average:', average)
    return average


def calculate_variance(image_1, image_2, n):
    """
    Calculate variance for two image sets.
    """
    variance = [((i - j) - (k - l)) ** 2 / 2 * (n - 1) for i, j, k, l in
                zip(image_1, np.mean(image_1, axis=(1, 2)), image_2, np.mean(image_2, axis=(1, 2)))]
    variance = np.array([np.mean(var) for var in variance])
    print('Variance:', variance)
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

    print(f"Linear fit parameters: slope = {popt_1[0]:.6f}, intercept = {popt_1[1]:.6f}")
    print(f"Gain: {gain:.3f} ± {gain_uncertainty:.3f} e-/ADU")
    return popt_1, gain


def plot_ptc_curve(average, variance, gain, popt_1, max_linear=55000):
    """
    Plot the PTC curve.
    """
    # Identify the saturation point
    max_index = np.argmax(variance)
    saturation_grey_value = average[max_index]
    variance_sqr_saturation_grey = variance[max_index]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(average, variance, 'ro')
    mask = (average < max_linear)
    x_fit = average[mask]
    ax.plot(x_fit, ptc_fit_high(x_fit, *popt_1), 'b-', label=rf'Gain = {gain:.3f} ' 'e$^\mathdefault{-}$ADU$^{-1}$')
    ax.plot(saturation_grey_value, variance_sqr_saturation_grey, 'b*', label=f'Well Depth = {saturation_grey_value * gain:4.0f} ' 'e$^\mathdefault{-}$')
    ax.set_xlabel('Pixel count (ADU)')
    ax.set_ylabel('Variance (ADU$^\mathdefault{2}$)')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()


def calculate_gain_standard_deviation(n_pixels):
    """
    Calculate the standard deviation of the gain.
    """
    sigma_gain = 100 * np.sqrt(8 / n_pixels)
    return sigma_gain


def save_results(average, variance, gain, popt_1, sigma_gain, x_max, path):
    """
    Save PTC results to a configuration file.
    """
    config = configparser.ConfigParser()

    config['PTC'] = {
        'Gain': str(popt_1[0]),
        'High Gain': str(1 / popt_1[0]),
        'Standard Deviation of Gain': str(sigma_gain),
        'Average': ', '.join(map(str, average)),
        'Variance': ', '.join(map(str, variance)),
        'FWC': str(x_max),
        'FWC e-': str(x_max * gain),
    }

    with open(os.path.join(path, 'ptc_config.ini'), 'w') as configfile:
        config.write(configfile)


def calculate_ptc(path, save_path, n_pixels=1024 * 1280, n=2):
    """
    Main function to calculate and save the PTC.
    """
    image_1, image_2, exposures = load_image_data(path)
    average = calculate_average(image_1, image_2)
    variance = calculate_variance(image_1, image_2, n)
    popt_1, gain = fit_ptc_curve(average, variance)

    max_index = np.argmax(variance)
    x_max = average[max_index]
    saturation_value = x_max

    plot_ptc_curve(average, variance, gain, popt_1)

    sigma_gain = calculate_gain_standard_deviation(n_pixels)
    print('The slope for high Gain is:', popt_1)
    print('The standard deviation of the gain is:', sigma_gain, '%')

    save_results(average, variance, gain, popt_1, sigma_gain, x_max, save_path)
    return saturation_value, average, variance, exposures


# Set paths
path = '/Volumes/SanDisk-2TB-SSD/w1m/ptc'
save_path = '/Volumes/SanDisk-2TB-SSD/w1m/'

# Calculate PTC
saturation_value, average, variance, exposures = calculate_ptc(path, save_path)

def plot_linearity_line(gradient, offset, startx, endx, step, figure, ax1):
    plt.figure(figure)
    x_values = []
    y_values = []

    for x in np.arange(startx, endx, step):
        y = x * gradient + offset  # y = mx + c
        x_values.append(x)
        y_values.append(y)
    ax1.plot(x_values, y_values, 'b-', label=f'({gradient:.1f} ' r'$\mathdefault{\mathrm{s}^{-1} \times t_\mathrm{exp} + }$' f'{offset:.1f}) ADU')


def plot_linearity(exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, corrected_counts, figure):
    startx = (min(ExposureTimeList_5_95))
    endx = (max(ExposureTimeList_5_95))
    step = 0.0001

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], wspace=0, hspace=0)
    figure = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_ylabel('Signal (ADU)')
    ax1.plot([exposure_times], [corrected_counts], 'ro')
    plot_linearity_line(Linearitygradient, LinearityOffset, startx, endx, step, figure, ax1)
    ax1.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax1.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    ax1.legend(loc='lower right')

    ax2.plot(exposure_times, ResidualsList, 'ro', linewidth=1)
    ax2.plot([startx, endx], [0, 0], 'b-', linewidth=1)
    ax2.set_ylim(-3 * LinearityError, +3 * LinearityError)
    ax2.set_ylabel('Residuals (\%)')
    ax2.set_xlabel('Exposure (s)')
    ax2.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax2.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    # figure.savefig('Linearity_FFR_12bit_temp_-25.pdf', bbox_inches='tight')
    plt.show()


def set_best_fit_ranges(xdata, ydata, startx, endx):
    print(xdata, ydata)
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

    print('gradient [{}] offset [{}]'.format(Gradient, Offset))
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
        print('creating residuals')
        for i in range(0, len(self.y)):
            calculated_level = self.offset + (self.gradient * self.x[i])

            Residuals = (self.y[i] - calculated_level) / (
                    self.range * self.max) * 100  # Equation 35 from EMVA1288-V3.1

            residualscts = (self.y[i] - calculated_level)

            self.residuals.append(Residuals)
            self.residual_counts.append(residualscts)

        print('residuals [{}]'.format(self.residuals))
        print('residual counts [{}]'.format(self.residual_counts))


def find_linearity_error(ResidualsList):
    LinearityError = (max(ResidualsList) - min(ResidualsList)) / 2

    return LinearityError


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

plot_linearity(exposures, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
               ResidualsList_5_95, LinearityError, ResidualsList, average, figure)