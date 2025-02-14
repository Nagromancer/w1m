from path import Path
import os
import sys
import traceback
import warnings
import tempfile
import subprocess
import argparse as ap
import sep
import numpy as np
import astropy.units as u
import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from matplotlib import pyplot as plt
from utilities import plate_scale

warnings.filterwarnings("ignore")


def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser("Solve astrometry for a set of images")
    p.add_argument('cat_file',
                   help='Master fits catalog',
                   type=str)
    p.add_argument('input_images',
                   help='Input images to solve with given catalog',
                   type=str,
                   nargs='+',
                   default=[])
    p.add_argument('binning',
                   help='Binning of the detector',
                   type=int,)
    p.add_argument('camera',
                   type=str,
                   choices=['red', 'blue'],
                   help='Camera colour.')
    p.add_argument('--indir',
                   help='location of input files',
                   default='.',
                   type=str)
    p.add_argument('--outdir',
                   help='location of output files',
                   default='.',
                   type=str)
    p.add_argument('--force3rd',
                   help='force a 3rd order distortion polyfit',
                   action='store_true',
                   default=False)
    p.add_argument('--save_matched_cat',
                   help='output the matched catalog with basic photometry',
                   action='store_true',
                   default=False)
    p.add_argument('--plot', help='plot the residuals', action='store_true', default=False)
    return p.parse_args()


def _detect_objects_sep(data, background_rms, area_min, area_max,
                        detection_sigma, trim_border=10):
    """
    Find objects in an image array using SEP

    Parameters
    ----------
    data : array
        Image array to source detect on
    background_rms
        Std of the sky background
    area_min : int
        Minimum number of pixels for an object to be valid
    area_max : int
        Maximum number of pixels for an object to be valid
    detection_sigma : float
        Number of sigma above the background for source detecting
    trim_border : int
        Number of pixels to exclude from the edge of the image array

    Returns
    -------
    objects : astropy Table
        A list of detected objects in astropy Table format

    Raises
    ------
    None
    """

    raw_objects = sep.extract(data, detection_sigma * background_rms, minarea=area_min)

    raw_objects = Table(raw_objects[np.logical_and.reduce([
        raw_objects['npix'] < area_max,
        # Filter targets near the edge of the frame
        raw_objects['xmin'] > trim_border,
        raw_objects['xmax'] < data.shape[1] - trim_border,
        raw_objects['ymin'] > trim_border,
        raw_objects['ymax'] < data.shape[0] - trim_border
    ])])

    # Astrometry.net expects 1-index pixel positions
    objects = Table()
    objects['X'] = raw_objects['x'] + 1
    objects['Y'] = raw_objects['y'] + 1
    objects['FLUX'] = raw_objects['cflux']
    objects.sort('FLUX')
    objects.reverse()

    return objects


def _wcs_from_table(objects, frame_shape, scale_low, scale_high, estimate_coord=None,
                    estimate_coord_radius=None, timeout=120):
    """
    Attempt to calculate a WCS solution for a given table of object detections.

    Parameters
    ----------
    table : Astropy Table
        Contains columns X, Y, FLUX, sorted by FLUX descending
    frame_shape : array
        array of frame [height, width]
    scale_low : float
        Minimum plate scale in arcsecond/px
    scale_high : float
        Maximum plate scale in arcsecond/px
    estimate_coord : SkyCoord
        Estimated position of the field
    estimate_coord_radius : float
        Radius to search around estimated coords
    timeout : int
        Abort if processing takes this long
        Default 25

    Parameters
    ----------
    solution : dict
        Dictionary of WCS header keywords

    Raises
    ------
    None
    """
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            xyls_path = os.path.join(tempdir, 'scratch.xyls')
            wcs_path = os.path.join(tempdir, 'scratch.wcs')
            objects.write(xyls_path, format='fits')

            astrometry_args = [
                '/opt/homebrew/bin/solve-field',
                '--no-plots',
                '--scale-units', 'arcsecperpix',
                '-t 3', '-l 3', '--no-remove-lines',  # '--no-fits2fits',
                '--scale-high', str(scale_high), '--scale-low', str(scale_low),
                '--width', str(frame_shape[1]), '--height', str(frame_shape[0]),
                xyls_path]

            if estimate_coord is not None and estimate_coord_radius is not None:
                astrometry_args += [
                    '--ra', str(estimate_coord.ra.to_value(u.deg)),
                    '--dec', str(estimate_coord.dec.to_value(u.deg)),
                    '--radius', str(estimate_coord_radius.to_value(u.deg)),
                ]

            # print(f"Running: {' '.join(astrometry_args)}")
            subprocess.check_call(astrometry_args, cwd=tempdir,
                                  timeout=timeout,
                                  stdout=subprocess.DEVNULL,  # Suppresses standard output
                                  stderr=subprocess.DEVNULL)  # Suppresses standard error


            wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
            solution = {}
            try:
                with open(wcs_path) as wcs_file:
                    header = wcs_file.read()
                    # ds9 will only accept newline-delimited keys
                    # so we need to reformat the 80-character cards
                    for line in [header[i:i + 80] for i in range(0, len(header), 80)]:
                        key = line[0:8].strip()
                        if '=' in line and key not in wcs_ignore_cards:
                            card = fits.Card.fromstring(line)
                            solution[card.keyword] = card.value
                return solution
            except FileNotFoundError:
                print('Failed to find WCS solution')
                return {}

    except Exception:
        print('Failed to update wcs with error:')
        traceback.print_exc(file=sys.stdout)
        return {}


def check_wcs_corners(wcs_header, objects, catalog, frame_shape, check_tile_size=512):
    """
    Sanity check the WCS solution in the corners of the detector

    Parameters
    ----------
    wcs_header : dict
        Dictionary containing WCS solution
    objects : Sep Objects
        Sep output for source detection
    catalog : Astropy Table
        Catalog od objects for comparing
    frame_shape : tuple
        Shape of the detector

    Returns
    -------
    header : dict
        Updated fits header with corners + centre performance

    Raises
    ------
    None
    """
    tile_check_regions = {
        'MATCH-TL': [0, check_tile_size, frame_shape[0] - check_tile_size, frame_shape[0]],
        'MATCH-TR': [frame_shape[1] - check_tile_size, frame_shape[1], frame_shape[0] - check_tile_size,
                     frame_shape[0]],
        'MATCH-BL': [0, check_tile_size, 0, check_tile_size],
        'MATCH-BR': [frame_shape[1] - check_tile_size, frame_shape[1], 0, check_tile_size],
        'MATCH-C': [(frame_shape[1] - check_tile_size) // 2, (frame_shape[1] + check_tile_size) // 2,
                    (frame_shape[0] - check_tile_size) // 2, (frame_shape[0] + check_tile_size) // 2]
    }

    wcs_x, wcs_y = WCS(wcs_header).all_world2pix(catalog['RA'], catalog['DEC'], 1)
    delta_x = np.abs(wcs_x - objects['X'])
    delta_y = np.abs(wcs_y - objects['Y'])
    delta_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)

    header = {}
    for k, v in tile_check_regions.items():
        check_objects = np.logical_and.reduce([
            objects['X'] > v[0],
            objects['X'] < v[1],
            objects['Y'] > v[2],
            objects['Y'] < v[3],
        ])

        median = np.median(delta_xy[check_objects])
        header[k] = -1 if np.isnan(median) else median

    return header, delta_xy


def fit_hdu_distortion(wcs_header, objects, catalog, force3rd):
    """
    Fit the image distortion parameters

    Parameters
    ----------
    wcs_header : dict
        WCS solution
    objects : Sep Objects
        Catalog of source detections
    catalog : Astropy Table
        Astrometric catalog
    force3rd : boolean
        Force a new 3rd order polynomial

    Returns
    -------
    fitted_header : dict
        Updated fits header with WCS solution

    Raises
    ------
    None
    """
    # The SIP paper's U and V coordinates are found by applying the core (i.e. CD matrix)
    # transformation to the RA and Dec, ignoring distortion; relative to CRPIX
    wcs = WCS(wcs_header)
    U, V = wcs.wcs_world2pix(catalog['RA'], catalog['DEC'], 1)
    U -= wcs_header['CRPIX1']
    V -= wcs_header['CRPIX2']

    # The SIP paper's u and v coordinates are simply the image coordinates relative to CRPIX
    u = objects['X'] - wcs_header['CRPIX1']
    v = objects['Y'] - wcs_header['CRPIX2']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Solve for f(u, v) = U - u
        f_init = polynomial_from_header(wcs_header, 'A', force3rd)
        f_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        f_poly, _ = f_fit(f_init, u, v, U - u)

        # Solve for g(u, v) = V - v
        g_init = polynomial_from_header(wcs_header, 'B', force3rd)
        g_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        g_poly, _ = g_fit(g_init, u, v, V - v)

    # Return a copy of the header with updated distortion coefficients
    fitted_header = wcs_header.copy()

    for c, a, b in zip(f_poly.param_names, f_poly.parameters, g_poly.parameters):
        fitted_header['A_' + c[1:]] = a
        fitted_header['B_' + c[1:]] = b

    fitted_header['A_ORDER'] = f_init.degree
    fitted_header['B_ORDER'] = g_init.degree
    fitted_header['CTYPE1'] = 'RA---TAN-SIP'
    fitted_header['CTYPE2'] = 'DEC--TAN-SIP'

    return fitted_header


def polynomial_from_header(wcs_header, axis, force3rd):
    """
    Get polynomial from the header

    Parameters
    ----------
    wcs_header : dict
        WCS solution
    axis : int
        Order of axis
    force3rd : boolean
        Force a new 3rd order polynomial

    Returns
    -------
    2D polynomial model

    Raises
    ------
    None
    """
    # Astrometry.net sometimes doesn't fit distortion terms!
    if axis + '_ORDER' not in wcs_header or force3rd:
        return models.Polynomial2D(degree=3)

    coeffs = {}
    for key in wcs_header:
        if key.startswith(axis + '_') and key != axis + '_ORDER':
            coeffs['c' + key[2:]] = wcs_header[key]

    return models.Polynomial2D(degree=wcs_header[axis + '_ORDER'], **coeffs)


def prepare_frame(input_path, output_path, catalog, force3rd, reference_mag):
    """
    Prepare the frame for WCS solution. The output is the solved image

    Parameters
    ----------
    input_path : string
        Name of input file
    output_path : string
        Name of output file
    catalog : astropy Table
        Reference catalog
    force3rd : boolean
        Force a new 3rd order poly for distortion fitting
    reference_magnitude : string
        Reference magnitude colour for the catalog

    Returns
    -------
    None

    Raises
    ------
    None
    """
    frame = fits.open(input_path)[0]
    try:
        frame_exptime = float(frame.header['EXPOSURE'])
    except KeyError:
        frame_exptime = float(frame.header['EXPTIME'])

    catalog = catalog[catalog['VALID'] == True]
    gids = catalog['ID']
    colour_mag = catalog[reference_mag]
    # make a cross-match mask
    cm_mask = np.where(((~np.isnan(gids)) & (colour_mag <= 17) & (colour_mag >= 12)))[0]
    # make a trimmed catalog for cross-matching
    catalog_cm = catalog[cm_mask]

    # Prepare image
    frame_data = frame.data.astype(float)
    frame_bg = sep.Background(frame_data)
    frame_data_corr = frame_data - frame_bg

    # save the image with the same format as the input
    output = fits.PrimaryHDU(frame_data.astype(np.float32), header=frame.header)

    area_min = 10
    area_max = 400
    scale_min = plate_scale * args.binning * 0.95
    scale_max = plate_scale * args.binning * 1.05
    detection_sigma = 3
    zp_clip_sigma = 3

    try:
        estimate_coord = SkyCoord(ra=frame.header['TELRA'],
                                  dec=frame.header['TELDEC'],
                                  unit=(u.hourangle, u.deg))
    except KeyError:
        try:
            estimate_coord = SkyCoord(ra=frame.header['MNTRAD'],
                                      dec=frame.header['MNTDECD'],
                                      unit=(u.deg, u.deg))
        except KeyError:
            print('No commanded RA position, skipping!')
            return None, None, None, None, None

    estimate_coord_radius = 1 * u.deg

    # Detect all objects in the image and attempt a full-frame solution
    objects = _detect_objects_sep(frame_data_corr, frame_bg.globalrms, area_min,
                                  area_max, detection_sigma)
    # print(f"Found {len(objects)} objects in {input_path}")

    wcs_header = _wcs_from_table(objects, frame_data.shape, scale_min,
                                 scale_max, estimate_coord, estimate_coord_radius)
    initial_wcs = WCS(wcs_header)

    if not wcs_header:
        print('Failed to find initial WCS solution - aborting')
        return None, None, None, None, None

    # if it fails, skip the image and stick it in a bad folder
    try:
        object_ra, object_dec = WCS(wcs_header).all_pix2world(objects['X'],
                                                              objects['Y'], 1,
                                                              ra_dec_order=True)
        objects['RA'] = object_ra
        objects['DEC'] = object_dec

        cat_coords = SkyCoord(ra=catalog_cm['RA'] * u.degree,
                              dec=catalog_cm['DEC'] * u.degree)

        # Iteratively improve the cross-match, WCS fit, and ZP estimation
        i = 0
        while True:
            # Cross-match vs the catalog so we can exclude false detections and improve our distortion fit
            object_coordinates = SkyCoord(ra=objects['RA'] * u.degree, dec=objects['DEC'] * u.degree)
            match_idx, _, _ = object_coordinates.match_to_catalog_sky(cat_coords)
            matched_cat = catalog_cm[match_idx]

            # add check here for matches, try to catch bad catalog
            # print("Number of matches (pre-distance cut-off):  {}".format(len(match_idx)))

            wcs_x, wcs_y = WCS(wcs_header).all_world2pix(matched_cat['RA'],
                                                         matched_cat['DEC'], 1)
            delta_x = np.abs(wcs_x - objects['X'])
            delta_y = np.abs(wcs_y - objects['Y'])
            delta_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)

            zp_mask = delta_xy > 10  # pixels

            zp_delta_mag = matched_cat[reference_mag] + 2.5 * np.log10(objects['FLUX'] / frame_exptime)
            zp_mean, _, zp_stddev = sigma_clipped_stats(zp_delta_mag, mask=zp_mask, sigma=zp_clip_sigma)

            # Discard blends and any objects with inconsistent brightness
            zp_filter = np.logical_and.reduce([
                np.logical_not(zp_mask),
                zp_delta_mag > zp_mean - zp_clip_sigma * zp_stddev,
                zp_delta_mag < zp_mean + zp_clip_sigma * zp_stddev])

            # add check here for matches, try to catch bad catalog
            # print("Number of matches (post-distance cut-off): {}".format(len(delta_xy[zp_filter])))
            # print(f"Median delta_xy: {np.median(delta_xy[zp_filter]):.4f} pix")

            before_match, _ = check_wcs_corners(wcs_header, objects[zp_filter],
                                                matched_cat[zp_filter], frame_data.shape)

            wcs_header = fit_hdu_distortion(wcs_header, objects[zp_filter], matched_cat[zp_filter],
                                            force3rd)

            i += 1

            # Check for convergence
            objects['RA'], objects['DEC'] = WCS(wcs_header).all_pix2world(objects['X'],
                                                                          objects['Y'], 1,
                                                                          ra_dec_order=True)

            after_match, _ = check_wcs_corners(wcs_header, objects[zp_filter],
                                               matched_cat[zp_filter], frame_data.shape)
            match_improvement = np.max([before_match[x] - after_match[x] for x in after_match])
            if i > 5 or match_improvement < 0.01:
                break

        updated_wcs_header, xy_residuals = check_wcs_corners(wcs_header, objects[zp_filter],
                                                             matched_cat[zp_filter], frame_data.shape)

        # update the header from the final fit
        wcs_header.update(updated_wcs_header)

        for k, v in wcs_header.items():
            output.header[k] = v
        hdu_list = [output]

        # calculate flux with aperture radius of 20 pixels
        zp_calc_objects = objects[zp_filter]
        flux20, fluxerr20, _ = sep.sum_circle(frame_data_corr, zp_calc_objects['X'], zp_calc_objects['Y'], 20.0 / args.binning,
                                              subpix=0, gain=1)
        zp_calc_objects['FLUX20'] = flux20
        zp_calc_objects['FLUXERR20'] = fluxerr20
        zp_delta_mag = matched_cat[zp_filter][reference_mag] + 2.5 * np.log10(zp_calc_objects['FLUX20'] / frame_exptime)
        zp_mean, _, zp_stddev = sigma_clipped_stats(zp_delta_mag, sigma=zp_clip_sigma)

    except Exception:
        print("{} failed WCS, skipping...\n".format(input_path))
        # traceback.print_exc(file=sys.stdout)
        return None, None, None, None, None

    # Add zero point to header
    output.header['ZP_20r'] = zp_mean
    output.header['ZPSTD_20r'] = zp_stddev

    # Add background level and RMS to header
    output.header['BACK-LVL'] = frame_bg.globalback
    output.header['BACK-RMS'] = frame_bg.globalrms

    # print(f"Mean ZP (ADU) (10 pixel radius) = {zp_mean:.4f} ± {zp_stddev:.4f}, "
    #       f"Background Level = {frame_bg.globalback:.4f} ± {frame_bg.globalrms:.4f}")

    # output the updated solved fits image
    fits.HDUList(hdu_list).writeto(output_path, overwrite=True)

    return wcs_header, objects[zp_filter], matched_cat[zp_filter], xy_residuals, initial_wcs


if __name__ == "__main__":
    args = arg_parse()
    reference_mag = "BP_MAG" if args.camera == "blue" else "RP_MAG"

    # check for a catalog
    if not os.path.exists(args.cat_file):
        print("{} is missing, skipping...".format(args.cat_file))
        sys.exit(1)

    # we want to make a catalog that matches the source detect closely
    master_catalog = Table.read(args.cat_file)

    # store the WCS headers for the final check if objects appear on chip
    # at least once during the list of given reference images
    wcs_store = []
    print(f"Solving astrometry for {len(args.input_images)} images...")
    for input_image in tqdm.tqdm(args.input_images):
        base_name = input_image.split(".fits")[0]
        if os.path.exists(input_image) and os.path.exists(args.outdir):
            header = fits.getheader(input_image)

            # check for existing solved images by looking for A_0_0
            if 'A_0_0' in header and ('ZP_20r' in header or 'ZP_20R' in header):
                # print(f"{input_image} already solved, skipping...")
                wcs_store.append(WCS(header))
                continue

            x_width = header['NAXIS1']
            y_width = header['NAXIS2']
            final_wcs, objects_matched, catalog_matched, residuals, initial_wcs = prepare_frame(input_image,
                                                                                   input_image,
                                                                                   master_catalog,
                                                                                   args.force3rd,
                                                                                   reference_mag)
            wcs_store.append(final_wcs)

            if final_wcs is None:
                print("Failed to solve {}".format(input_image))
                continue

            if args.plot:
                plt.rcParams['font.family'] = 'Times'

                # plot diagnostics on the fitting here using the matched catalog/objects
                # from the source detection
                ref_x = final_wcs['CRPIX1']
                ref_y = final_wcs['CRPIX2']
                radial_distance = np.sqrt((np.array(objects_matched['X']) - ref_x) ** 2 +
                                          (np.array(objects_matched['Y']) - ref_y) ** 2)

                # generate vecotrs for quiver plot
                vector_scale = 1000
                wcs_pos_x, wcs_pos_y = WCS(final_wcs).all_world2pix(catalog_matched['RA'],
                                                                    catalog_matched['DEC'], 1)
                x_comp = (wcs_pos_x - objects_matched['X'])
                y_comp = (wcs_pos_y - objects_matched['Y'])


                init_pos_x, init_pos_y = initial_wcs.all_world2pix(catalog_matched['RA'],
                                                                   catalog_matched['DEC'], 1)
                x_comp_init = (init_pos_x - objects_matched['X'])
                y_comp_init = (init_pos_y - objects_matched['Y'])
                residuals_init = np.sqrt(x_comp_init ** 2 + y_comp_init ** 2)
                print(f"Initial median residuals: {np.median(residuals_init):.4f} pix")
                print(f"Fitted median residuals: {np.median(residuals):.4f} pix")

                fig_q, ax_q = plt.subplots(2, figsize=(10, 20))
                ax_q[0].set_title('WCS - Source Detect Positions (x{})'.format(vector_scale))
                ax_q[0].quiver(objects_matched['X'], objects_matched['Y'], x_comp_init, y_comp_init, units='xy', scale_units='xy',
                               scale=1 / vector_scale)
                ax_q[0].set_xlim(0, x_width)
                ax_q[0].set_ylim(0, y_width)
                ax_q[1].set_title('WCS - Source Detect Positions (x{})'.format(vector_scale))
                ax_q[1].quiver(objects_matched['X'], objects_matched['Y'], x_comp, y_comp, units='xy', scale_units='xy',
                               scale=1 / vector_scale)
                ax_q[1].set_xlim(0, x_width)
                ax_q[1].set_ylim(0, y_width)
                fig_q.tight_layout()
                fig_q.savefig('{}_quiver_plot.png'.format(base_name))

                fig, ax = plt.subplots(2, figsize=(10, 20))
                # plot a zoom into the residuals to 2 pixels
                ax[0].plot(radial_distance, residuals, 'k.')
                ax[0].set_xlabel('Radial position from CRPIX (pix)')
                ax[0].set_ylabel('Delta XY (pix)')

                # plot residuals versus brightness
                ax[1].plot(catalog_matched[reference_mag], residuals, 'k.')
                if args.camera == 'blue':
                    ax[1].set_xlabel('${G}_\mathrm{BP}$ (mag)')
                else:
                    ax[1].set_xlabel('${G}_\mathrm{RP}$ (mag)')
                ax[1].set_ylabel('Delta XY (pix)')

                fig.tight_layout()
                fig.savefig('{}_wcs_residuals.png'.format(base_name))
        else:
            print(f"Missing input or output file, skipping...: {input_image} or {args.outdir}")
    # check if at least one reference image passed
    if len(args.input_images) > wcs_store.count(None):
        sys.exit(0)
    else:
        print('No solved reference images')
        sys.exit(1)
