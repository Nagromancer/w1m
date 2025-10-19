# bash script to sort files by camera colour and target

home_dir="/Volumes/SanDisk-2TB-SSD/w1m"
bin="/Users/nagro/PycharmProjects/w1m"

dates=("$home_dir"/dates/*) # Expand correctly
first_date="20251014"

new_dates=()
found_first=false
for date in "${dates[@]}"; do
  dir_name=$(basename "$date" | tr -d '[:space:]') # Trim spaces/newlines
  if [[ "$dir_name" == "$first_date" ]]; then
    found_first=true
  fi
  if $found_first; then
    new_dates+=("$date")
  fi
done

dates=("${new_dates[@]}") # Overwrite the original dates array with filtered values

for date in $dates; do
  echo $date
done

for date in $dates; do
  base_dir=$date

  mkdir -p $base_dir/flat
  for file in $base_dir/*; do
        if [[ $file == *"-flat-"* ]]; then
          mv $file $base_dir/$cam/flat
        fi
      done

  date_only=$(basename $base_dir)
  echo "Processing $date_only"

    # target name is given by the first part of the file name - everything before the second to last dash
    # create a directory for each target in the red and blue directories
    # first to see if there are any files ending with .fits in the directory
    if ls $base_dir/*.fits 1> /dev/null 2>&1; then
      for file in $base_dir/*".fits"; do
#      if [[ $file == *"flat"* ]]; then continue; fi # don't treat the flat directory as a target
# do for dark and bias files too
      if [[ $file == *"dark"* || $file == *"bias"* || $file == *"flat"* ]]; then
        continue
      fi
      target=$(awk -F'-' '{
      if (NF >= 2) {
        for (i = 1; i <= NF - 2; i++) {
          printf "%s%s", $i, (i == NF - 2 ? "" : "-")
        }
      } else {
        print $0
      }
    }' <<<"$file")
      mkdir -p $target/raw
      mv $file $target/raw
    done
    fi

  # activate conda environment
  eval "$(conda shell.bash hook)"
  conda activate globular

  # calibrate the raw files
  echo "Calibrating raw files"
  # use fitsheader to get the binning of the flat files - just use one of the flat files
  flat_file=$(ls $base_dir/flat/*BLUE*.fits | head -n 1)

  bin_x=$(fitsheader $flat_file | grep "XBIN" | awk '{print $2}')
  bin_y=$(fitsheader $flat_file | grep "YBIN" | awk '{print $2}')
  echo "Detected binning: $bin_x x $bin_y in the flat files"

  # check to see if master flat already exists
  if [[ ! -f $base_dir/flat/master-flat-blue.fits ]]; then
    echo "Creating master flat for blue camera with binning $bin_x x $bin_y"
    python $bin/flats.py $base_dir/flat $home_dir/calibration_frames/master-bias-blue-"$bin_x".fits $home_dir/calibration_frames/master-dark-blue-"$bin_x".fits $base_dir/flat
  else
    echo "Master flat already exists for blue camera with binning $bin_x x $bin_y"
  fi

  if [[ -d $target/plots ]]; then
    echo "Skipping $target"
    continue
  fi

  # get a list of targets from the non-flat directories in the base directory
  targets=$(find $base_dir -mindepth 1 -maxdepth 1 -type d ! -name "flat")

  for target in $targets; do
    target=$(realpath "$target") # get the absolute path of the target
    if [[ $target == *"flat"* ]]; then continue; fi # don't treat the flat directory as a target
    target_only=$(basename $target)
    echo "Processing $target_only on $date_only"

    raw_file=$(ls $target/raw/*.fits | head -n 1)
    bin_x=$(fitsheader $raw_file | grep "XBIN" | awk '{print $2}')
    bin_y=$(fitsheader $raw_file | grep "YBIN" | awk '{print $2}')
    echo "Detected binning: $bin_x x $bin_y in the raw files"

    # make calibrated directory and calibrate images
    mkdir -p $target/calibrated
    python $bin/calibrate.py $target/raw $home_dir/calibration_frames/master-bias-blue-"$bin_x".fits $home_dir/calibration_frames/master-dark-blue-"$bin_x".fits $base_dir/flat/master-flat-blue.fits $target/calibrated
    if [[ $? -ne 0 ]]; then
      echo "Error processing $target"
      exit 1
    fi

    # check to see if reference catalogue exists for this target
    ref_cat="$home_dir/reference_catalogues/$target_only/$date_only-$target_only.fits"
    if [[ ! -f $ref_cat ]]; then
      mkdir -p $home_dir/reference_catalogues/$target_only
      python $bin/fetch_reference_catalogue.py 0.27 0.27 "$ref_cat" --img_path $target/raw --target "$target_only"
      if [[ $? -ne 0 ]]; then
        echo "Error processing $target"
        exit 1
      fi
    fi

    # solve astrometry
    python $bin/solve-astrometry.py "$ref_cat" $target/calibrated/*.fits "$bin_x"
    if [[ $? -ne 0 ]]; then
      echo "Error processing $target. Failed to solve astrometry."
      exit 1
    fi

    # measure HFD
    python $bin/measure_hfd.py $target/calibrated "$bin_x"
    if [[ $? -ne 0 ]]; then
      echo "Error processing $target. Failed to measure HFD."
      exit 1
    fi
    # reject bad images
    mkdir -p $target/rejected
    python $bin/remove_bad_files.py $target/calibrated $target/rejected
    if [[ $? -ne 0 ]]; then
      echo "Error processing $target. Failed to reject bad files."
      exit 1
    fi
    # create diagnostic plots
    # check to see if diagnostic plots have already been created
    if [[ ! -d $target/plots ]]; then
      mkdir -p $target/plots
      python $bin/diagnostic_plots.py $target/calibrated $target/rejected $target/plots "$bin_x"
    fi
    if [[ $? -ne 0 ]]; then
      echo "Error processing $target. Failed to create diagnostic plots"
      exit 1
    fi

    # extract photometry
    python $bin/extract_photometry.py "$ref_cat" $target/calibrated "$target/$date_only-$target_only-phot.fits" $target/plots/lightcurve.png "$bin_x"
    echo "\n"
  done
done
