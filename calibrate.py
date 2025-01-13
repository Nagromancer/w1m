import warnings
import fitsio
import numpy as np
from tqdm import tqdm
from path import Path
import argparse

parser = argparse.ArgumentParser(description='Calibrate science images.')
parser.add_argument('raw_dir', type=str, help='Base directory containing raw science images.')
parser.add_argument('bias_path', type=str, help='Path to master bias image.')
parser.add_argument('dark_path', type=str, help='Path to master dark image.')
parser.add_argument('flat_path', type=str, help='Path to master flat-field image.')
parser.add_argument('out_dir', type=str, help='Output directory for calibrated science images.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing calibrated images.')
args = parser.parse_args()

warnings.filterwarnings("ignore")
raw_dir = Path(args.raw_dir)
out_dir = Path(args.out_dir)
bias_path = Path(args.bias_path)
dark_path = Path(args.dark_path)
flat_path = Path(args.flat_path)

if not out_dir.exists():
    out_dir.mkdir()

out_dir_files = out_dir.files()
out_dir_files = [file for file in out_dir_files if 'calibrated' in file.name]
if len(out_dir_files) > 0 and not args.overwrite:
    print(f"{len(out_dir_files)} calibrated images already exist in {out_dir}. Try --overwrite.")
    exit(42)


science_files = raw_dir.files()
science_files = [file for file in science_files if 'dark' not in file.name and 'flat' not in file.name and 'bias' not in file.name and '._' not in file.name and '.fits' in file.name]

science_files.sort()

master_bias = fitsio.read(bias_path)
master_dark = fitsio.read(dark_path)
master_flat = fitsio.read(flat_path)

dark_header = fitsio.read_header(dark_path)
dark_exp_time = dark_header['EXPTIME']
print(f"Calibrating {len(science_files)} science images...")

region = np.array([[int(x.replace("[", "").replace("]", "")) - 1 for x in y] for y in
                  [x.split(':') for x in (fitsio.read_header(science_files[0])['IMAG-RGN'].split(','))]])
region[:, 1] += 1
for file in tqdm(science_files):
    header = fitsio.read_header(file)
    try:
        data = fitsio.read(file).astype(np.int32)
    except OSError:
        print(f"Error reading {file}")
        continue
    data = data[region[1][0]:region[1][1], region[0][0]:region[0][1]]
    data = (data - master_bias - (master_dark * header['EXPTIME'] / dark_exp_time))
    data /= master_flat
    data = data.astype(np.float64)
    resolution = np.shape(data)

    data = data.astype(np.float32)

    header['HISTORY'] = 'Calibrated with master bias, dark, and flat-field images.'
    header['BZERO'] = 0
    fitsio.write(out_dir / f'calibrated-{file.name}', data, header=header, clobber=True)
