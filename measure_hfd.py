import fitsio
from astropy.table import Table
import sep
import tqdm
import numpy as np
from path import Path
import os
import sys
from astropy.wcs import WCS
import warnings
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.utils import calc_total_error
from utilities import plate_scale, red_gain, blue_gain
import argparse


def measure_hfd(files, camera, plot=True, sep_threshold=1.3, verbose=False):
    failed_extractions = 0
    failed_writes = 0
    failed_reads = 0

    gain = red_gain if camera == 'red' else blue_gain

    for file in tqdm.tqdm(files, file=sys.stdout):
        try:
            header = fitsio.read_header(file)
            if 'HFD' in header:
                # continue
                pass
            data = fitsio.read(file)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(file)

        except OSError:
            print(f"Error reading {file}. Removing from list.")
            os.remove(file)
            failed_reads += 1
            continue

        if verbose:
            print(f"Extracting objects from {file}")

        data = data.astype(np.float32)
        bkg = sep.Background(data)
        data_sub = data - bkg

        try:
            objects = Table(sep.extract(data_sub, sep_threshold, err=bkg.globalrms, gain=gain))
        except Exception:
            try:
                objects = Table(sep.extract(data_sub, sep_threshold + 2, err=bkg.globalrms, gain=gain))
            except Exception:
                failed_extractions += 1
                continue
        objects['theta'][objects['theta'] > np.pi / 2] -= np.pi / 2

        labels = ['x', 'y', 'tnpix', 'a', 'b', 'theta', 'flux']
        objects = Table([objects[label] for label in labels], names=labels)

        if verbose:
            print(f"Extracted {len(objects)} objects from {file.basename()}")

        ra, dec = wcs.all_pix2world(objects['x'], objects['y'], 0)
        objects['ra'] = ra
        objects['dec'] = dec

        ap = CircularAperture(np.transpose([objects['x'], objects['y']]), 5)
        error = calc_total_error(data, bkg.globalrms, gain)

        frame_apertures = aperture_photometry(data, ap, error=error, method='subpixel', subpixels=5)
        objects[f'FLUX_5'] = frame_apertures['aperture_sum']
        objects[f'FLUXERR_5'] = frame_apertures['aperture_sum_err']
        objects[f'SNR'] = objects['FLUX_5'] / objects['FLUXERR_5']
        objects = objects[1 - objects['b'] / objects['a'] < 0.5]  # ellipticity < 0.5
        objects = objects[objects['SNR'] > 5.0]
        objects = objects[objects['tnpix'] > 5]

        hfd_measurement_objects = objects[objects['flux'] > 10000.0]
        kronrad, kron_flag = sep.kron_radius(data_sub, hfd_measurement_objects['x'], hfd_measurement_objects['y'],
                                             hfd_measurement_objects['a'], hfd_measurement_objects['b'],
                                             hfd_measurement_objects['theta'], 6.0)
        flux, _, flux_flag = sep.sum_ellipse(data_sub, hfd_measurement_objects['x'], hfd_measurement_objects['y'],
                                             hfd_measurement_objects['a'], hfd_measurement_objects['b'],
                                             hfd_measurement_objects['theta'], 2.5 * kronrad, subpix=0)
        r, radius_flag = sep.flux_radius(data_sub, hfd_measurement_objects['x'], hfd_measurement_objects['y'],
                                         6.0 * hfd_measurement_objects['a'], 0.5, normflux=flux, subpix=5)
        filt = np.logical_and.reduce([
            kron_flag == 0,
            flux_flag == 0,
            radius_flag == 0
        ])
        fluxes = flux[filt]
        hfds1 = 2 * r[filt] * plate_scale
        if plot:
            # plot hfd against flux with log scale histogram of hfds
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # plot side by side
            ax[0].scatter(fluxes, hfds1, s=1)
            ax[0].set_xlabel('Flux')
            ax[0].set_ylabel('HFD')
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_title('HFD vs Flux')
            ax[1].hist(hfds1[hfds1 < 10], bins=100, log=True)
            ax[1].set_xlabel('HFD')
            ax[1].set_ylabel('Count')
            ax[1].set_title('HFD Histogram')
            ax[1].axvline(np.median(hfds1), color='r', linestyle='--')
            plt.tight_layout()
            plt.show()

        median_hfd = np.median(hfds1)

        # add hfd to header, with units in the comment
        header.add_record(dict(name='HFD', value=median_hfd, comment='Half-Flux Diameter (arcsec)'))

        try:
            fitsio.write(file, data, header=header, clobber=True)
        except OSError:
            failed_writes += 1
            continue

    print(f"Failed extractions: {failed_extractions}")
    print(f"Failed reads: {failed_reads}")
    print(f"Failed writes: {failed_writes}")


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Analyse background-subtracted images')
    parser.add_argument('input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('analysis_dir', type=str, help='Directory for the analysis files.')
    parser.add_argument('camera', type=str, choices=['red', 'blue'], help='Camera colour.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing analysis files.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.files())
    measure_hfd(image_files, camera=args.camera, plot=False)


if __name__ == '__main__':
    main()