from scipy import signal
from scipy.integrate import simps
import scipy
import numpy as np


DELTA_LOW = 1
DELTA_HIGH = 4

THETA_LOW = 6
THETA_HIGH = 10

# check these constants, might no longer be correct
FACTOR = 30
FREQ_BINS = 501
fs = 1000
NPERSEG = 1000


def resampling(lfp_signal, factor=FACTOR):
    size = len(lfp_signal)

    print('Starting resampling...')
    data_resampled = signal.resample(lfp_signal, round(size / factor))

    return data_resampled


def filter_design(order, fs):
    """
    Creates a filter given an order and sampling frequency, returns the filter in sos
    """
    sos = signal.iirfilter(order, [48, 52], btype='bandstop', analog=False, ftype='butter', fs=fs, output='sos')
    return sos


def power_calc(psd, freqs, band_low, band_high):
    """
    Calculates band power given edges of bands and power spectrum of an epoch.
    Effectively calculates area under the curve of a power graph.
    """
    idx_band = np.logical_and(freqs >= band_low, freqs <= band_high)

    freq_res = freqs[1] - freqs[0]
    band_power = simps(psd[idx_band], dx=freq_res)

    return band_power


def epoch_welch(epochs, fs, nperseg, bands=[(DELTA_LOW, DELTA_HIGH), (THETA_LOW, THETA_HIGH)]):
    """
    Performs a power spectrum density analysis on epochs given the sampling frequency
    and length of segment to use for the Welch's method.

    Returns frequency bins, powers within those bins and band_power.
    Band power is an array with shape (n_epochs, n_bands) starting with delta, followed by theta.
    """
    n_epochs = len(epochs)
    powers = np.zeros((n_epochs, FREQ_BINS))
    band_power = np.zeros((n_epochs, len(bands)))
    for i in range(n_epochs):
        power_spectrum = scipy.signal.welch(epochs[i], fs=fs, nperseg=nperseg)
        powers[i, :] = power_spectrum[1]

        # should pull this out the loop
        frequencies = power_spectrum[0]

        # extracts delta and theta band power: see power_calc for more.
        for j in range(len(bands)):
            band_power[i, j] = power_calc(power_spectrum[1], frequencies, bands[j][0], bands[j][1])

    print('Power Spectrum Density analysis completed.')
    return frequencies, powers, band_power


def analyse(data, loaded=False, resampled=False):
    """"
    Performs all analysis needed given a filepath.
    loaded indicates whether filepath is a string or an array of points.
    """
    # creates and applies filter
    sos = filter_design(17, fs=fs)
    data_filter = scipy.signal.sosfilt(sos, data_downsampled)
    print('Filter applied')

    # calculates power in delta/theta band for each epoch
    freqs, powers, power_per_band = epoch_welch(epochs, fs=fs, nperseg=NPERSEG)

    return power_per_band


def channel_average(filepaths):
    """
    Takes a list of filepaths and averages them into a single signal.
    """
    data = 0
    print("Starting averaging of channels...")

    # maybe even resample before averaging? Would save some RAM possibly
    # loops over all given files
    for files in filepaths:
        data_loaded = load(files)
        data = data_loaded['data'] + data

    # averages the signal
    data = data / len(filepaths)
    return data