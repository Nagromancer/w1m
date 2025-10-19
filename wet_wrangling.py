from path import Path
import fitsio

date = 20251015

# set up paths
base_dir = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}")
dark_dir = base_dir / "dark"
bias_dir = base_dir / "bias"
flat_dir = base_dir / "flat"
science_dir = base_dir / "science"

# create directories if they don't exist
for directory in [dark_dir, bias_dir, flat_dir, science_dir]:
    if not directory.exists():
        directory.mkdir()

# common metadata
phot_filter = "bg40"
observer = "Morgan Mitchell"
telcode = "warw"
object_name = "G2938"

# move files to appropriate directories
for file in base_dir.files("*.fits"):
    if "dark" in file.name.lower():
        file.move(dark_dir / file.name)
    elif "bias" in file.name.lower():
        file.move(bias_dir / file.name)
    elif "flat" in file.name.lower():
        file.move(flat_dir / file.name)
    elif "Gaia" in file.name:
        file.move(science_dir / file.name)

# rename files in science directory
for i, file in enumerate(sorted(science_dir.files("*.fits"))):
    if "BLUE" in file.name:
        new_name = f"{telcode}{date}_{object_name}_{phot_filter}_{i+1:04d}.fits"
        file.rename(science_dir / new_name)

# rename files in flat directory
for i, file in enumerate(sorted(flat_dir.files("*.fits"))):
    if "BLUE" in file.name:
        new_name = f"{telcode}{date}_flat_{phot_filter}_{i+1:04d}.fits"
        file.rename(flat_dir / new_name)

# rename files in dark directory
for i, file in enumerate(sorted(dark_dir.files("*.fits"))):
    if "BLUE" in file.name:
        new_name = f"{telcode}{date}_dark_{i+1:04d}.fits"
        file.rename(dark_dir / new_name)

# rename files in bias directory
for i, file in enumerate(sorted(bias_dir.files("*.fits"))):
    if "BLUE" in file.name:
        new_name = f"{telcode}{date}_bias_{i+1:04d}.fits"
        file.rename(bias_dir / new_name)

# add OBSERVER to headers
for directory in [dark_dir, bias_dir, flat_dir, science_dir]:
    for file in directory.files("*.fits"):
        try:
            header = fitsio.read_header(file)
            if header.get('OBSERVER', None) != observer:
                header['OBSERVER'] = observer
                fitsio.write(file, fitsio.read(file), header=header, clobber=True)
        except OSError:
            print(f"Error reading {file}, skipping.")

# move all the files back to the parent folder and delete the empty directories
for directory in [dark_dir, bias_dir, flat_dir, science_dir]:
    for file in directory.files("*.fits"):
        file.move(base_dir / file.name)
    directory.rmdir()