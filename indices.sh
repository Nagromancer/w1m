# shell script to download index files from astrometry.net
out_dir=/opt/homebrew/Cellar/astrometry-net/0.96/data

# we need the 5203 and 5204 index files, e.g.
# https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5203-00.fits
# https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5204-00.fits

for i in 5203 5204; do
    for j in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 35 39 43 47; do
        url=https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-${i}-${j}.fits
        out_file=${out_dir}/index-${i}-${j}.fits
        echo "Downloading ${url} to ${out_file}"
        curl -o ${out_file} ${url}
    done
done