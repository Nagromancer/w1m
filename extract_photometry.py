import numpy as np
from astropy.wcs import WCS
import argparse
import sep
from astropy.table import Table
from astropy.io import fits
from path import Path
from astropy.coordinates import SkyCoord
import warnings
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.time import Time
import astropy.units as u
import astropy.coordinates as coord

from utilities import blue_gain
from astropy.table import hstack, vstack
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32
warnings.filterwarnings("ignore")


def main(args):
    output_path = Path(args.output_path)
    plot_path = Path(args.plot_path)

    if "Gaia_DR3_" in str(output_path):
        target_id = int(output_path.name.split("Gaia_DR3_")[-1].replace("-phot.fits", ""))
    elif "Gaia DR3 " in str(output_path):
        target_id = int(output_path.name.split("Gaia DR3 ")[-1].replace("-phot.fits", ""))
    else:
        target_id = None
    print(f"Target ID: {target_id} detected from output path.")

    if not output_path.exists():
        print(f"Extracting photometry...")

        # get target name
        gain = blue_gain

        la_palma = coord.EarthLocation(lat=28.760255 * u.deg, lon=-17.879284 * u.deg, height=2396 * u.m)

        # get star properties for target
        cat_path = Path(args.cat_path)
        cat = Table.read(cat_path)

        # get image paths
        img_dir = Path(args.img_dir)
        image_files = sorted(img_dir.files("*.fits"))
        num_images = len(image_files)

        # we want to measure the flux of every star with every aperture radius, store the results in a table
        phot_table = Table()

        # loop over images
        for image_file in tqdm(image_files):

            # open image
            aperture_radii = [5, 10, 15, 20]  # unbinned pixels
            with fits.open(image_file) as hdul:
                frame = hdul[0]
                data = frame.data.astype(float)
                wcs = WCS(frame.header)
                exposure = frame.header["EXPTIME"] * u.s

                # convert catalogue coordinates to pixel coordinates
                sky_coords = SkyCoord(ra=cat["RA"], dec=cat["DEC"], unit="deg")
                pixel_coords = np.array(wcs.world_to_pixel(sky_coords))
                field_centre = wcs.pixel_to_world(data.shape[1] / 2, data.shape[0] / 2)

                frame_time = Time(frame.header["DATE-OBS"], format="isot", scale="utc") + exposure / 2
                ltt_bjd = frame_time.light_travel_time(field_centre, 'barycentric', location=la_palma)
                ltt_hjd = frame_time.light_travel_time(field_centre, 'heliocentric', location=la_palma)
                jd = frame_time.jd
                bjd = (frame_time.tdb + ltt_bjd).jd
                hjd = (frame_time.utc + ltt_hjd).jd

                # check if pixel coordinates are within image bounds
                valid_coords_idx = (pixel_coords[0] > (25 / args.binning)) & (pixel_coords[0] < (data.shape[1] - (25 / args.binning))) & (pixel_coords[1] > (25 / args.binning)) & (pixel_coords[1] < (data.shape[0] - (25 / args.binning)))
                pixel_coords = pixel_coords[:, valid_coords_idx]
                image_cat = cat[valid_coords_idx]

                # extract photometry
                frame_bg = sep.Background(data, bw=128 / args.binning, bh=128 / args.binning)
                frame_bg_rms = frame_bg.rms()
                frame_data_corr = data - frame_bg

                # create a table just for this image
                phot_table_i = Table()

                # add time columns
                phot_table_i["ID"] = image_cat["ID"]
                phot_table_i["JD"] = [jd] * len(image_cat)
                phot_table_i["HJD"] = [hjd] * len(image_cat)
                phot_table_i["BJD"] = [bjd] * len(image_cat)

                for r in aperture_radii:
                    zp = frame.header.get(f"ZP_{r}R_E-", 0)  # zero point in mag
                    ap = CircularAperture(pixel_coords.T, r=r / args.binning)
                    # calculate total error
                    error = np.sqrt(frame_bg_rms**2 + frame_data_corr / gain)
                    phot_table_i_r = aperture_photometry(frame_data_corr, ap, error=error)
                    # add units of electrons/s

                    phot_table_i_r["aperture_sum"] *= gain / exposure.value
                    phot_table_i_r["aperture_sum_err"] *= gain / exposure.value

                    # rename columns to reflect aperture radius
                    phot_table_i_r.rename_column("aperture_sum", f"FLUX_{r}")
                    phot_table_i_r.rename_column("aperture_sum_err", f"FLUX_ERR_{r}")


                    # add magnitude as well
                    phot_table_i_r[f"MAG_{r}"] = -2.5 * np.log10(phot_table_i_r[f"FLUX_{r}"]) + zp
                    phot_table_i_r[f"MAG_ERR_{r}"] = np.abs(phot_table_i_r[f"MAG_{r}"] - (-2.5 * np.log10(phot_table_i_r[f"FLUX_{r}"] + phot_table_i_r[f"FLUX_ERR_{r}"]) + zp))

                    # add units of electrons/s
                    phot_table_i_r[f"FLUX_{r}"].unit = u.electron / u.s
                    phot_table_i_r[f"FLUX_ERR_{r}"].unit = u.electron / u.s
                    phot_table_i_r[f"MAG_{r}"].unit = u.mag
                    phot_table_i_r[f"MAG_ERR_{r}"].unit = u.mag

                    # add flux and flux error to the table_i (ONLY these columns)
                    phot_table_i = hstack([phot_table_i, phot_table_i_r[[f"FLUX_{r}", f"FLUX_ERR_{r}", f"MAG_{r}", f"MAG_ERR_{r}"]]])

                phot_table = vstack([phot_table, phot_table_i])
        # save photometry table
        cat["MEASUREMENTS"] = np.zeros(len(cat), dtype=int)
        for i, star in enumerate(cat):
            cat["MEASUREMENTS"][i] = len(phot_table[phot_table["ID"] == star["ID"]])
        cat = cat[cat["MEASUREMENTS"] == num_images]
        # correct catalogue by including only stars with as many measurements as there are images
        phot_table = phot_table[np.isin(np.array(phot_table["ID"]), np.array(cat["ID"]))]
        phot_table.write(output_path, overwrite=True)
    else:
        phot_table = Table.read(output_path)

    if not plot_path.exists() and target_id is not None:
        print(f"Plot path exists? {plot_path.exists()}")
        target_table = phot_table[phot_table["ID"] == target_id]
        min_time = np.min(target_table["BJD"])
        # find the smallest multiple of 100 less than min_time to use as the offset
        offset = (min_time // 100) * 100
        # plt.errorbar(np.array(target_table["BJD"]) - offset, np.array(target_table["MAG_5"]), yerr=np.array(target_table["MAG_ERR_5"]), fmt='o', color='red', capsize=5)
        # plt.errorbar(np.array(target_table["BJD"]) - offset, np.array(target_table["MAG_10"]), yerr=np.array(target_table["MAG_ERR_10"]), fmt='o', color='black', capsize=5)
        plt.errorbar(np.array(target_table["BJD"]) - offset, np.array(target_table["MAG_15"]), yerr=np.array(target_table["MAG_ERR_15"]), fmt='o', color='green', capsize=5)
        # plt.errorbar(np.array(target_table["BJD"]) - offset, np.array(target_table["MAG_20"]), yerr=np.array(target_table["MAG_ERR_20"]), fmt='o', color='orange', capsize=5)
        plt.xlabel(r'Time (BJD$_\mathrm{TDB}$'f' - {offset:.0f})')
        plt.ylabel('$G_{\mathrm{BP}}$ (mag)')
        plt.gca().invert_yaxis()  # invert y-axis for magnitude
        # put gaia in italics
        plt.title(fr'\textit{{Gaia}} DR3 {target_id}')
        plt.grid()
        plt.savefig(plot_path)
        print(f"Saved light curve for target {target_id} at {plot_path}.")
        plt.close()
    else:
        print(f"Light curve for target {target_id} already exists at {plot_path}.")
        print(f"Plot path exists? {plot_path.exists()}")
        print(f"ID: {target_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Photometry')
    parser.add_argument('cat_path', type=str, help='Path to the catalogue.')
    parser.add_argument('img_dir', type=str, help='Base directory containing the images.')
    parser.add_argument('output_path', type=str, help='Output directory for photometry table.')
    parser.add_argument('plot_path', type=str, help='Output directory for plots.')
    parser.add_argument('binning', type=int, help='Binning factor.')
    args = parser.parse_args()
    main(args)
