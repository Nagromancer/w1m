import warnings
from path import Path
import matplotlib.pyplot as plt
from diagnostic_plots import read_headers, extract_times
import numpy as np
import matplotlib as mpl


def extract_focus(headers):
    focuses = []
    for header in headers:
        focuses.append(round(header.get('TELFOCUS', np.nan), 0))
    focuses = np.array(focuses)
    return focuses


def extract_ha_dec(headers):
    has = []
    decs = []
    for header in headers:
        has.append(header.get('TELHAD', np.nan))
        decs.append(header.get('TELDECD', np.nan))
    has = np.array(has)
    decs = np.array(decs)
    return has, decs


def extract_fds(headers, flux_fractions):
    fds = []
    for header in headers:
        try:
            fd = []
            for flux_fraction in flux_fractions:
                fd.append(header[f'FD{int(flux_fraction * 100):02d}'])
            fds.append(np.array(fd))
        except KeyError:
            fds.append([np.nan] * len(flux_fractions))
    return np.array(fds)


def main():
    warnings.filterwarnings("ignore")

    date = 20250902
    input_dir = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/Gaia_DR3_380316777081879680/calibrated")
    reject_dir = Path(f"/Volumes/SanDisk-2TB-SSD/w1m/dates/{date}/Gaia_DR3_380316777081879680/rejected")
    output_dir = Path(f"{date}_metadata.csv")

    headers = read_headers(input_dir, reject_dir)
    if len(headers) == 0:
        print("No headers found in input directories.")
        return

    flux_fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                      0.95]

    times = extract_times(headers)
    times = np.array(times)
    focuses = extract_focus(headers)
    fds = extract_fds(headers, flux_fractions)
    has, decs = extract_ha_dec(headers)

    # fd10 = fds[:, 1]
    # fd90 = fds[:, -2]
    # plt.plot(times, fd90 / fd10, '.')
    # plt.xlabel('Time (UTC)')
    # plt.ylabel('Flux Diameter Ratio (90\% / 10\%)')
    # plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    # plt.title(date)
    # plt.grid()
    # plt.show()

    # Save to CSV
    data = np.column_stack((times, focuses, has, decs))

    # add the flux fractions to data
    data = np.column_stack((data, fds))

    header = "TIME,FOCUS,HA,DEC," + ",".join([f"FD{int(ff*100):02d}" for ff in flux_fractions])
    np.savetxt(output_dir, data, delimiter=",", header=header, comments='', fmt=['%s', '%.0f', '%.5f', '%.5f'] + ['%.3f'] * len(flux_fractions))



if __name__ == '__main__':
    main()