import numpy as np
import warnings
import matplotlib.pyplot as plt
from path import Path
import fitsio

warnings.filterwarnings("ignore", category=RuntimeWarning)
base_path = Path("/Volumes/SanDisk-2TB-SSD/w1m/dates")

# quick month dictionary
month_no_to_name = {
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Aug",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec"
}


# w1m data
w1m_lc_path = Path("/Users/nagro/PycharmProjects/w1m/lcs/w1m_detrended_flux.csv")
w1m_lc = np.genfromtxt(w1m_lc_path, delimiter=",", names=True)
w1m_time = w1m_lc["bjd"]
w1m_flux = w1m_lc["flux"]
w1m_flux_err = w1m_lc["err"]

# TEMPORARY
flux_err_nights = np.split(w1m_flux_err, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
flux_nights = np.split(w1m_flux, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
time_nights = np.split(w1m_time, np.where(np.diff(w1m_time) > 0.5)[0] + 1)
mid_obs_nights = np.array([np.median(t) for t in time_nights])
# round to nearest integer
mid_obs_nights = np.round(mid_obs_nights - 0.5).astype(int)

# convert the BJDs into calendar dates
import astropy.time
calendar_dates = astropy.time.Time(mid_obs_nights, format='jd').to_value('iso', subfmt='date')
calendar_dates = [str(date).replace("-", "") for date in calendar_dates]

for date in calendar_dates:
    path = base_path / date / "blue" / "Gaia DR3 1571584539980588544" / "raw"
    if not path.exists():
        path = base_path / date / "blue" / "Gaia_DR3_1571584539980588544" / "raw"
    raw_files = sorted(path.files("*.fits"))

    # format date instead of 20250205 as 2025 Feb 05
    formatted_date = f"{date[:4]} {month_no_to_name[date[4:6]]} {date[6:]}"

    # get the start and end times from the headers
    header_start = fitsio.read_header(raw_files[0])
    start_time = header_start["DATE-OBS"]
    start_time = astropy.time.Time(start_time, format='isot')
    start_time = str(start_time).replace("T", " ")[11:16]

    header_end = fitsio.read_header(raw_files[-1])
    end_time = header_end["DATE-END"]
    end_time = astropy.time.Time(end_time, format='isot')
    end_time = end_time.utc.iso.replace("T", " ")[11:16]

    print(f"{formatted_date} & {start_time}$-${end_time} & BG40 & 30 & {len(raw_files)} \\\\")


