import fitsio
import tqdm
import numpy as np
from path import Path
import os
import sys
from astropy.wcs import WCS
import warnings
import argparse


def check_images(files, output_path, min_zp=23, max_hfd=4):
    count = 0
    for file in tqdm.tqdm(files, file=sys.stdout):
        header = fitsio.read_header(file)

        # check for wcs solution by looking for A_0_0 in header
        if 'A_0_0' in header and 'HFD' in header and 'ZP_10R' in header:
            continue

        if header['HFD'] < max_hfd and header['ZP_10R'] > min_zp:
            continue

        print(f"Moving {file} to bad files directory.")
        os.rename(file, output_path / file.name)
        count += 1
    print(f"Moved {count} files to bad files directory ({count / len(files) * 100:.2f}%) to {output_path}")


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Analyse background-subtracted images')
    parser.add_argument('input_dir', type=str, help='Directory containing input images.')
    parser.add_argument('output_dir', type=str, help='Directory to move the bad files to.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.files())
    output_path = Path(args.output_dir)
    check_images(image_files, output_path)


if __name__ == '__main__':
    main()