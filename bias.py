import os
import warnings
from logging import raiseExceptions

import fitsio
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create master bias image.')
parser.add_argument('raw_dir', type=str, help='Base directory containing raw bias images.')
parser.add_argument('out_dir', type=str, help='Output directory for master bias image.')
args = parser.parse_args()

warnings.filterwarnings("ignore")
raw_dir = Path(args.raw_dir)
out_dir = Path(args.out_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

bias_files = raw_dir.files('*.fits')
bias_files = [file for file in bias_files if not "._" in str(file)]  # remove hidden files
if len(bias_files) == 0:
    print("No bias images found.")
    exit(1)
bias_files.sort()
colour = "blue" if "BLUE" in str(bias_files[0]) else "red" if "RED" in str(bias_files[0]) else raiseExceptions

# get shape from IMAG-RGN element of header e.g. [21:2068, 31:2048] and convert from 1-indexed to 0-indexed
region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                  [x.split(':') for x in (fitsio.read_header(bias_files[0])['IMAG-RGN'].split(','))]])
region[:, 1] += 1
shape = (region[:, 1] - region[:, 0])[1], (region[:, 1] - region[:, 0])[0], len(bias_files)

stacked_biases = np.zeros(shape, dtype=np.float32)
for i, file in enumerate(tqdm(bias_files)):
    stacked_biases[:, :, i] = fitsio.read(file)[region[1][0]:region[1][1], region[0][0]:region[0][1]]

# perform a trimmed mean to remove outliers - ignore highest and lowest values
stacked_biases = np.median(stacked_biases, axis=2)
master_bias = stacked_biases.astype(np.float32)

print(f"Master bias resolution: {master_bias.shape[0]} x {master_bias.shape[1]}")
print(f"Median of master bias: {np.median(master_bias):.4f}")

fitsio.write(out_dir / f'master-bias-{colour}.fits', master_bias, clobber=True)
print(f"Master bias image saved to {out_dir / f'master-bias-{colour}.fits'}")
