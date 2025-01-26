import os
import warnings
from logging import raiseExceptions

import fitsio
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create master flat-field image.')
parser.add_argument('raw_dir', type=str, help='Base directory containing raw flat-field images.')
parser.add_argument('bias_path', type=str, help='Path to master bias image.')
parser.add_argument('dark_path', type=str, help='Path to master dark image.')
parser.add_argument('out_dir', type=str, help='Output directory for master flat-field image.')
args = parser.parse_args()

warnings.filterwarnings("ignore")
raw_dir = Path(args.raw_dir)
bias_path = Path(args.bias_path)
dark_path = Path(args.dark_path)
out_dir = Path(args.out_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

flat_files = raw_dir.files('*flat*')
flat_files = [file for file in flat_files if "._" not in str(file)]  # remove hidden files
if len(flat_files) == 0:
    print("No flat-field images found.")
    exit(1)
if not bias_path.exists():
    print("Master bias image not found.")
    exit(1)
if not dark_path.exists():
    print("Master dark image not found.")
    exit(1)
flat_files.sort()
colour = "blue" if "BLUE" in str(flat_files[0]) else "red" if "RED" in str(flat_files[0]) else raiseExceptions
out_path = out_dir / f'master-flat-{colour}.fits'
if out_path.exists():
    print(f"Master flat-field image already exists at {out_path}.")
    exit(1)

master_bias = fitsio.read(bias_path)
master_dark = fitsio.read(dark_path)
flat_cube = None

region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                   [x.split(':') for x in (fitsio.read_header(flat_files[0])['IMAG-RGN'].split(','))]])
region[:, 1] += 1

print(f"Stacking {len(flat_files)} flat-field images...")
count = 0
for file in tqdm(flat_files):
    exposure_time = fitsio.read_header(file)['EXPTIME']
    moon_sep = fitsio.read_header(file)['MOONSEP']
    if moon_sep < 30:
        print(f"Skipping flat-field image {file} with moon separation {moon_sep:.1f} degrees.")
        continue
    count += 1
    dark_exposure_time = fitsio.read_header(dark_path)['EXPTIME']
    data = fitsio.read(file)
    data = data[region[1][0]:region[1][1], region[0][0]:region[0][1]].astype(np.float32) - master_bias - master_dark * exposure_time / dark_exposure_time
    data /= np.median(data)
    flat_cube = np.dstack((flat_cube, data)) if flat_cube is not None else data
stacked_flats = np.median(flat_cube, axis=2)

print(f"Stacked {count} flat-field images.")
print(f"Master flat resolution: {stacked_flats.shape[0]} x {stacked_flats.shape[1]}")

fitsio.write(out_path, stacked_flats, clobber=True)
print(f"Master flat-field image saved to {out_path}")
