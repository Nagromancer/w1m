import argparse
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from tqdm import tqdm
import fitsio
from path import Path


def plot_images():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times'
    plt.rcParams['font.size'] = 15

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times')


def read_bias_data(directory):
    list_images = list(directory.glob('*.fits'))
    bias_values = []

    print('Reading bias images...')
    for image_path in tqdm(list_images):
        header = fits.getheader(image_path)
        region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                           [x.split(':') for x in (header['IMAG-RGN'].split(','))]])
        region[:, 1] += 1
        hdulist = fits.open(image_path)
        image_data = hdulist[0].data[region[1][0]:region[1][1], region[0][0]:region[0][1]]
        hdulist.close()
        bias_values.append(image_data)

    bias_values = np.array(bias_values)

    mean_bias = np.mean(bias_values, axis=0).flatten()
    bias_std = np.std(bias_values, axis=0).flatten()
    return mean_bias, bias_std, directory


def plot_read_noise(mean_bias, bias_std, camera):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0, hspace=0)

    ax1 = plt.subplot(gs[0])
    # median of the means
    med_means = np.median(mean_bias)
    hb = ax1.hist2d(mean_bias, bias_std, bins=80, cmap='inferno_r', norm=LogNorm(), range=[[int(med_means)-20, int(med_means)+20], [0, 40]])
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('Standard Deviation (ADU)')

    ax2 = plt.subplot(gs[1])
    ax2.hist(bias_std, bins=80, orientation='horizontal', color='k', histtype='step', density=False, range=[0, 40])
    ax2.set_xlabel('Number of Pixels')
    ax2.set_xscale('log')
    # put ticks at 10^2 and 10^5
    ax2.set_xticks([1e1, 1e3, 1e5])
    ax2.set_xticks([1e0, 1e2, 1e4], minor=True)
    ax2.set_xticklabels([], minor=True)


    value_median_hist = np.median(bias_std)
    print(f'Median bias std-dev = {value_median_hist:.3f} ADU')
    mean_bias_hist = np.mean(bias_std)
    print(f'Mean bias std-dev   = {mean_bias_hist:.3f} ADU')
    RMS = np.sqrt(np.mean(bias_std ** 2))
    print(f'RMS bias std-dev    = {RMS:.3f} ADU')
    ax2.axhline(RMS, color='r', linestyle=':')
    ax2.yaxis.set_ticklabels([])
    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of pixels')
    fig.tight_layout()
    plt.savefig(f'/Volumes/SanDisk-2TB-SSD/w1m/rn-analysis/read_noise_{camera}.pdf')
    plt.show()


def main():
    plot_images()
    camera = "red"
    bias_dir = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/rn-analysis/{camera}/high")
    print(bias_dir)

    mean_bias, bias_std, directory = read_bias_data(bias_dir)
    plot_read_noise(mean_bias, bias_std, camera)


if __name__ == "__main__":
    main()