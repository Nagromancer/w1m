# bash script to sort files by camera colour and target

home_dir="/Volumes/SanDisk-2TB-SSD/w1m"
bin="/Users/nagro/PycharmProjects/w1m"

dates=($home_dir/dates/*)

for date in $dates; do
  base_dir=$date
  date_only=$(basename $base_dir)
  echo "Processing $date_only"

  # check to see if base_dir/red or base_dir/blue exists do not exist
  if [[ ! -d $base_dir/red ]] || [[ ! -d $base_dir/blue ]]; then
    # create a red and blue directory
    mkdir -p $base_dir/red
    mkdir -p $base_dir/blue

    # move red and blue files to the appropriate directory
    for file in $base_dir/*; do
        if [[ $file == *"RED"* ]]; then
            mv $file $base_dir/red
        elif [[ $file == *"BLUE"* ]]; then
            mv $file $base_dir/blue
        fi
    done

    # create a flat directory in the red and blue directories
    mkdir -p $base_dir/red/flat
    mkdir -p $base_dir/blue/flat

    # move flat files to the appropriate directory
    for cam in blue red; do
      for file in $base_dir/$cam/*; do
          if [[ $file == *"-flat-"* ]]; then
              mv $file $base_dir/$cam/flat
          fi
      done
    done

    # target name is given by the first part of the file name - everything before the second to last dash
    # create a directory for each target in the red and blue directories
    for cam in blue red; do
      for file in $base_dir/$cam/*; do
          target=$(echo $file | awk -F "-" '{print $1"-"$2"-"$3}')
          if [[ $target == *"flat"* ]]; then continue; fi  # don't treat the flat directory as a target
          mkdir -p $target/raw
          mv $file $target/raw
      done
    done
  fi

  # activate conda environment
  eval "$(conda shell.bash hook)"
  conda activate globular

  # calibrate the raw files
  for cam in blue red; do
    echo "Processing $cam camera on $date_only"
    python $bin/flats.py $base_dir/$cam/flat $home_dir/calibration_frames/master-bias-$cam.fits $home_dir/calibration_frames/master-dark-$cam.fits $base_dir/$cam/flat
    for target in $base_dir/$cam/*; do
      if [[ -d $target/plots ]]; then
        echo "Skipping $target"
        continue
      fi
      # skip if the target has already been processed
        if [[ -d $target ]]; then
          if [[ $target == *"flat"* ]]; then continue; fi  # don't treat the flat directory as a target
          target_only=$(basename $target)
          echo "Processing $target_only on $cam camera on $date_only"

          # make calibrated directory and calibrate images
          mkdir -p $target/calibrated
          python $bin/calibrate.py $target/raw $home_dir/calibration_frames/master-bias-$cam.fits $home_dir/calibration_frames/master-dark-$cam.fits $base_dir/$cam/flat/master-flat-$cam.fits $target/calibrated
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
          python $bin/solve-astrometry.py "$ref_cat" $target/calibrated/*.fits
          if [[ $? -ne 0 ]]; then
            echo "Error processing $target"
            exit 1
          fi

          # measure HFD
          python $bin/measure_hfd.py $target/calibrated $cam
          if [[ $? -ne 0 ]]; then
            echo "Error processing $target"
            exit 1
          fi

          # reject bad images
          mkdir -p $target/rejected
          python $bin/remove_bad_files.py $target/calibrated $target/rejected
          if [[ $? -ne 0 ]]; then
            echo "Error processing $target"
            exit 1
          fi

          # create diagnostic plots
          # check to see if diagnostic plots have already been created
          if [[ ! -d $target/plots ]]; then
            mkdir -p $target/plots
            python $bin/diagnostic_plots.py $target/calibrated $target/rejected $target/plots
          fi
          if [[ $? -ne 0 ]]; then
            echo "Error processing $target"
            exit 1
          fi
        fi
    done
    echo "\n"
  done
done