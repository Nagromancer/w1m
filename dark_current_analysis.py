import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
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


def plot_dark_hist(dark_path, camera):
    data = fits.getdata(dark_path)
    data = data.flatten()
    # data = data[data > 0]  # remove negative values

    # print(f"Proportion of negative values: {1 - len(data) / data.size:.4f}")

    exposure = fits.getheader(dark_path)['EXPTIME']
    gain = 1.055 if camera == "blue" else 2.011  # e-/ADU
    data = data * gain / exposure  # e-/pixel/s
    print(f"Median dark current: {np.median(data):.4f} e-/pixel/s")

    fig, ax = plt.subplots()

    # logorithmically spaced bins
    bins = np.linspace(1, 10, 100) if camera == "red" else np.linspace(data.min(), 1, 100)

    ax.hist(data, bins=bins, histtype='step', color='black', lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Dark current (e$^{-}$px$^{-1}$s$^{-1}$)')
    ax.set_ylabel('Number of pixels')
    ax.axvline(np.median(data), color='red', linestyle='--', label=f'Median: {np.median(data):.4f} ' "e$^{-}$px$^{-1}$s$^{-1}$")
    ax.legend(loc='upper right')
    plt.savefig(f'/Volumes/SanDisk-2TB-SSD/w1m/calibration_frames/dark-current-histogram-{camera}-2.pdf', bbox_inches='tight')
    plt.show()


def main():
    plot_images()
    camera = "red"
    master_dark_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/calibration_frames/master-dark-{camera}-2.fits")
    plot_dark_hist(master_dark_path, camera)


if __name__ == "__main__":
    main()