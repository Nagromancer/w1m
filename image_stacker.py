from path import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import wotan
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from tqdm import tqdm
from astropy import wcs
from reproject import reproject_interp
import sep

plt.rcParams['figure.figsize'] = [14, 9]
plt.rcParams["font.family"] = "Times"
plt.rcParams["font.size"] = 32
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 32

# camera = "blue"
# dates = ["20250205", "20250206", "20250207", "20250208", "20250213", "20250216", "20250218", "20250219", "20250220", "20250221", "20250222", "20250223", "20250224", "20250225", "20250226", "20250310", "20250317", "20250318", "20250320", "20250321"]
# targ_id = 1571584539980588544
#
#
# images = []
# for date in dates:
#     try:
#         target = f"Gaia DR3 {targ_id}"
#         image_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/calibrated")
#         images += sorted(image_path.files("*.fits"))
#     except FileNotFoundError:
#         target = target.replace(" ", "_")
#         image_path = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/{camera}/{target}/calibrated")
#         images += sorted(image_path.files("*.fits"))
#
# print(f"Found {len(images)} images ({len(images) / 120:.2f} hrs)")
#
# hfds = []
# bg_rmss = []
# zps = []
# for image in tqdm(images):
#     header = fits.getheader(image)
#     hfds.append(header["HFD"])
#     bg_rmss.append(header["BACK-RMS"])
#     try:
#         zps.append(header["ZP_10R"])
#     except KeyError:
#         zps.append(header["ZP_20R"])
#
# bad_mask = np.zeros(len(hfds), dtype=bool)
# bad_mask[np.where(np.array(hfds) > 3)[0]] = True
# bad_mask[np.where(np.array(bg_rmss) > 30)[0]] = True
# bad_mask[np.where(np.array(zps) < 24)[0]] = True
#
# hfds = np.array(hfds)[~bad_mask]
# bg_rmss = np.array(bg_rmss)[~bad_mask]
# zps = np.array(zps)[~bad_mask]
#
# images = np.array(images)[~bad_mask]
#
# print(f"Using {len(hfds)} images ({len(hfds) / 120:.2f} hrs)")
#
# # background subtraction
# out_path = Path(f"/Volumes/SanDisk-2TB-SSD/stack")
# for image in tqdm(images):
#     image_data = fits.getdata(image).byteswap().newbyteorder()
#     header = fits.getheader(image)
#     new_image = out_path / f"{image.split('/')[-1]}"
#
#     frame_bg = sep.Background(image_data, bw=64, bh=64)
#     array_corr = image_data - frame_bg
#
#     fits.writeto(new_image, array_corr, header, overwrite=True)

aligned_path = Path(f"/Volumes/SanDisk-2TB-SSD/stack-shifted")
aligned_images = sorted(aligned_path.files("*.fits"))

# calculate a median stack
# would require too much memory to load all images at once
# so we will stack them in chunks
chunk_size = 100
num_chunks = len(aligned_images) // chunk_size + 1

intermediate_medians = np.zeros((num_chunks, 1023, 1023))

for i in tqdm(range(num_chunks)):
    chunk = aligned_images[i * chunk_size:(i + 1) * chunk_size]
    chunk_data = [fits.getdata(image) for image in chunk]
    chunk_data = np.array(chunk_data)
    intermediate_medians[i] = np.median(chunk_data, axis=0)

final_median = np.median(intermediate_medians, axis=0)
fits.writeto("/Volumes/SanDisk-2TB-SSD/median.fits", final_median, overwrite=True)
