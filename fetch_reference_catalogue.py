import argparse as ap
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import warnings

def generate_query_polygon(center, width, height, max_step_distance):
    """
    Generates an ADQL polygon that covers the box defined by
    center, width, and height, at a given step distance

    :param center: SkyCoord for the field center
    :param width: Field width (tangent plane)
    :param height: Field height
    :param max_step_distance: polygon point spacing (tangent plane)
    :return: SkyCoord containing sampling grid points
    """

    top = []
    bottom = []
    left = []
    right = []

    top_dec = center.dec + height / 2
    bottom_dec = center.dec - height / 2

    # Clip points below the pole (where we don't observe anyway)
    north_pole = 90 * u.degree
    if top_dec > north_pole:
        top.append(SkyCoord(0 * u.degree, 90 * u.degree))
    else:
        ra = center.ra - width / (2 * np.cos(top_dec))
        end_ra = center.ra + width / (2 * np.cos(top_dec))
        while ra < end_ra:
            top.append(SkyCoord(ra, top_dec))
            ra += max_step_distance / np.cos(top_dec)
        top.append(SkyCoord(end_ra, top_dec))

    ra = center.ra + width / (2 * np.cos(bottom_dec))
    end_ra = center.ra - width / (2 * np.cos(bottom_dec))
    while ra > end_ra:
        bottom.append(SkyCoord(ra, bottom_dec))
        ra -= max_step_distance / np.cos(bottom_dec)
    bottom.append(SkyCoord(end_ra, bottom_dec))

    dec = bottom_dec + max_step_distance
    while dec < top_dec:
        left.append(SkyCoord(center.ra - width / (2 * np.cos(dec)), dec))
        right.append(SkyCoord(center.ra + width / (2 * np.cos(dec)), dec))
        dec += max_step_distance

    right.reverse()
    return SkyCoord(top + right + bottom + left)


def query_tap(center, width, height, calibration_limits, blend_delta, step_distance=1 * u.degree):
    polygon_points = generate_query_polygon(center, width, height, step_distance)
    polygon_points = ', '.join(['{0}, {1}'.format(x.ra.to_value(u.degree), x.dec.to_value(u.degree)) for x in polygon_points])
    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_bp_n_obs, phot_rp_n_obs, phot_variable_flag \
        FROM gaiadr3.gaia_source \
        WHERE CONTAINS(
            POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec), \
            POLYGON('ICRS',{polygon_points}))=1 \
        AND phot_g_mean_mag < {calibration_limits[1] + blend_delta} and phot_g_mean_mag > {calibration_limits[0] - blend_delta};
    """

    from astroquery.gaia import Gaia
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    Gaia.remove_jobs([job.jobid])

    valid_reference = np.logical_and.reduce([
        results['phot_variable_flag'] != 'VARIABLE',
        results['phot_bp_n_obs'] > 0,
        results['phot_rp_n_obs'] > 0,
        results['phot_g_mean_mag'] < calibration_limits[1],
        results['phot_g_mean_mag'] > calibration_limits[0]])

    column_names = [
        'source_id', 'ra_deg', 'dec_deg', 'pmra', 'pmdec',
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_variable_flag',
        'valid_reference'
    ]

    column_dtype = [int] + [np.float64] * (len(column_names) - 3) + [str] + [bool]
    columns = [
        results['SOURCE_ID'],
        results['ra'] / u.degree,
        results['dec'] / u.degree,
        results['pmra'] * u.year / u.mas,
        results['pmdec'] * u.year / u.mas,
        results['phot_g_mean_mag'] / u.mag,
        results['phot_bp_mean_mag'] / u.mag,
        results['phot_rp_mean_mag'] / u.mag,
        results['phot_variable_flag'],
        valid_reference
    ]

    catalog = Table(columns, names=column_names, dtype=column_dtype)
    catalog['phot_variable_flag'] = catalog['phot_variable_flag'] == 'VARIABLE'

    # Fix invalid proper motion entries
    catalog['pmra'][np.isnan(catalog['pmra'])] = 0
    catalog['pmdec'][np.isnan(catalog['pmdec'])] = 0
    return catalog


def fetch_gaia(ra_center, dec_center, box_width, box_height, reference_epoch, output_path,
               calibration_limits, blend_radius, blend_delta):
    center = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg))

    start = Time.now()

    catalog = query_tap(center, box_width * u.degree, box_height * u.degree, calibration_limits, blend_delta)

    print(f'fetched in {(Time.now() - start).to(u.second):.3f}')

    # Apply proper motion offsets
    reference_epoch = Time(reference_epoch, scale='utc')
    delta_years = (reference_epoch - Time('2016-01-01T0:0:0')).to(u.year).value
    catalog['ra_corr'] = catalog['ra_deg'] + catalog['pmra'] * delta_years / (3.6e6 * np.cos(np.radians(catalog['dec_deg'])))
    catalog['dec_corr'] = catalog['dec_deg'] + catalog['pmdec'] * delta_years / 3.6e6

    print('Found {} candidate calibration stars'.format(np.sum(catalog['valid_reference'])))
    print('Checking for nearby blends...')
    start = Time.now()

    ra_exclusion = dec_exclusion = blend_radius

    # Improve performance by restricting the sources that we check
    # and by prefiltering stripes in RA
    reference_catalog = catalog[catalog['valid_reference']]
    reference_index = np.arange(len(catalog))[catalog['valid_reference']]
    for ra_start in range(0, 361):
        reference_mask = np.logical_and(
            reference_catalog['ra_corr'] > ra_start,
            reference_catalog['ra_corr'] < ra_start + 1
        )

        if not np.sum(reference_mask):
            continue

        check_catalog = catalog[np.logical_and(
            catalog['ra_corr'] > ra_start - ra_exclusion,
            catalog['ra_corr'] < ra_start + 1 + ra_exclusion
        )]

        # Account for sources on the other side of the 0h RA wrap
        if ra_start == 0:
            wrap_catalog = catalog[catalog['ra_corr'] > 360 - ra_exclusion]
            wrap_catalog['ra_corr'] -= 360
            check_catalog = vstack([check_catalog, wrap_catalog])
        elif ra_start == 359:
            wrap_catalog = catalog[catalog['ra_corr'] < ra_exclusion]
            wrap_catalog['ra_corr'] += 360
            check_catalog = vstack([check_catalog, wrap_catalog])

        for i in reference_index[reference_mask]:
            matches = np.logical_and.reduce([
                np.abs(check_catalog['ra_corr'] - catalog['ra_corr'][i]) < ra_exclusion,
                np.abs(check_catalog['dec_corr'] - catalog['dec_corr'][i]) < dec_exclusion,
                check_catalog['phot_g_mean_mag'] - catalog['phot_g_mean_mag'][i] < blend_delta
            ])

            catalog['valid_reference'][i] = np.sum(matches) == 1

    print(f'blend check in {(Time.now() - start).to(u.second):.3f}')
    print('Kept {} unblended calibration stars'.format(np.sum(catalog['valid_reference'])))

    meta = {
        'RA': center.ra.to_value(u.degree),
        'DEC': center.dec.to_value(u.degree),
        'WIDTH': box_width,
        'HEIGHT': box_height,
        'EPOCH': reference_epoch.isot,
        'MAGLIM': calibration_limits[1],
        'BLENDRAD': blend_radius,
        'BLENDMAG': blend_delta,
    }

    output = Table(catalog['source_id', 'ra_corr', 'dec_corr', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                           'valid_reference', 'pmra', 'pmdec', 'phot_variable_flag'],
                   names=['ID', 'RA', 'DEC', 'G_MAG', 'BP_MAG', 'RP_MAG', 'VALID', 'pmRA', 'pmDEC', 'VARIABLE'],
                   meta=meta)

    # add Bp-Rp color
    output['BP_RP'] = output['BP_MAG'] - output['RP_MAG']

    # plot the valid reference stars
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["figure.figsize"] = [10, 8]
    fig, ax = plt.subplots()
    ax.scatter(output['RA'], output['DEC'], s=np.sqrt(10**(-0.4 * output['G_MAG'])*1e7), c=output['VALID'])
    ax.set_xlim(center.ra.to_value(u.degree) - box_width / 2, center.ra.to_value(u.degree) + box_width / 2)
    ax.set_ylim(center.dec.to_value(u.degree) - box_height / 2, center.dec.to_value(u.degree) + box_height / 2)
    print(f"Max RA {center.ra.to_value(u.degree) + box_width / 2}")
    plt.show()

    output.sort('G_MAG')
    output[output['VALID']==True].pprint_all(max_width=-1, max_lines=-1)

    output.write(output_path, format='fits', overwrite=True)
    print(f'Wrote {len(output)} reference stars to {output_path}')


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('ra',
                        type=str,
                        help='Field center RA in degrees.')
    parser.add_argument('dec',
                        type=str,
                        help='Field center Dec in degrees.')
    parser.add_argument('box_width',
                        type=float,
                        metavar='box-width',
                        help='Box width in tangent-plane degrees.')
    parser.add_argument('box_height',
                        type=float,
                        metavar='box-height',
                        help='Box height in degrees.')
    parser.add_argument('output',
                        type=str,
                        default='.',
                        help='Path to save the output catalog.')
    parser.add_argument('--epoch',
                        type=str,
                        default='2025-01-01T0:0:0',
                        help='Reference datetime string for proper-motion calculations.')
    parser.add_argument('--calibration-limits',
                        metavar=('MIN', 'MAX'),
                        nargs=2,
                        type=float,
                        default=(10, 21),
                        help='Minimum and maximum G magnitude to consider for calibration stars')
    parser.add_argument('--blend-radius',
                        type=float,
                        default=0.00625,
                        help='Maximum cross-matched distance in degrees to be considered as blended')
    parser.add_argument('--blend-delta',
                        type=float,
                        default=3,
                        help='Maximum magnitude delta for a star to be considered as blended')

    args = parser.parse_args()
    fetch_gaia(args.ra, args.dec, args.box_width, args.box_height, args.epoch, args.output,
               args.calibration_limits, args.blend_radius, args.blend_delta)