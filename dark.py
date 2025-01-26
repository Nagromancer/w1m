import os
import warnings
from logging import raiseExceptions

import fitsio
import numpy as np
from path import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Create master dark image.')
parser.add_argument('raw_dir', type=str, help='Base directory containing the raw dark images.')
parser.add_argument('bias_path', type=str, help='Path to master bias image.')
parser.add_argument('out_dir', type=str, help='Output directory for master dark image.')
args = parser.parse_args()

warnings.filterwarnings("ignore")
raw_dir = Path(args.raw_dir)
out_dir = Path(args.out_dir)
bias_path = Path(args.bias_path)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

dark_files = raw_dir.files('dark*')
if len(dark_files) == 0:
    print("No dark-field images found.")
    exit(1)
dark_files.sort()
print("Fetching master bias image...")
colour = "blue" if "BLUE" in str(dark_files[0]) else "red" if "RED" in str(dark_files[0]) else raiseExceptions

out_path = out_dir / f'master-dark-{colour}.fits'
if out_path.exists():
    print(f"Master dark image already exists at {out_path}.")

master_bias = fitsio.read(bias_path)

# get shape from IMAG-RGN element of header e.g. [21:2068, 31:2048] and convert from 1-indexed to 0-indexed
region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                  [x.split(':') for x in (fitsio.read_header(dark_files[0])['IMAG-RGN'].split(','))]])
region[:, 1] += 1
shape = (region[:, 1] - region[:, 0])[1], (region[:, 1] - region[:, 0])[0], len(dark_files)

stacked_darks = np.zeros(shape, dtype=np.float32)
for i, file in enumerate(tqdm(dark_files)):
    stacked_darks[:, :, i] = fitsio.read(file)[region[1][0]:region[1][1], region[0][0]:region[0][1]]

stacked_darks = np.median(np.sort(stacked_darks, axis=2)[:, :, 1:-1], axis=2)
print(f"Median of stacked darks: {np.median(stacked_darks):.4f}")
print(f"Median of master bias (with bias): {np.median(master_bias):.4f}")
master_dark = stacked_darks.astype(np.float32) - master_bias
print(f"Median of master dark (without bias): {np.median(master_dark):.4f}")

# read exposure time from header
t = float(fitsio.read_header(dark_files[0])['EXPTIME'])
print(f"Exposure time: {t} s")
print(f"Median of master dark: {np.median(master_dark)/t:.4f} ADU/pixel/s")

# create new header containing exposure time
new_header = fitsio.FITSHDR()
new_header.add_record(dict(name='EXPTIME', value=t, comment='Exposure time (s)'))

fitsio.write(out_dir / out_path, master_dark, clobber=True, header=new_header)
print(f"Master dark image saved to {out_path}")
