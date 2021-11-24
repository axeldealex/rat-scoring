import pickle
"""
Created on Sun Aug  3 15:18:38 2014
@author: Dan Denman and Josh Siegle
Loads .continuous, .events, and .spikes files saved from the Open Ephys GUI
Usage:
    import OpenEphys
    data = OpenEphys.load(pathToFile) # returns a dict with data, timestamps, etc.
"""

import os
import numpy as np
import scipy.signal
import scipy.io
import time
import struct
from copy import deepcopy

# constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
BYTES_PER_SAMPLE = 2
RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + 10  # size of each continuous record in bytes
RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

# constants for pre-allocating matrices:
MAX_NUMBER_OF_SPIKES = int(1e6)
MAX_NUMBER_OF_RECORDS = int(1e6)
MAX_NUMBER_OF_EVENTS = int(1e6)


def load(filepath, dtype=float):
    # redirects to code for individual file types
    if 'continuous' in filepath:
        data = loadContinuous(filepath, dtype)
    elif 'spikes' in filepath:
        data = loadSpikes(filepath)
    elif 'events' in filepath:
        data = loadEvents(filepath)
    else:
        raise Exception("Not a recognized file type. Please input a .continuous, .spikes, or .events file")

    return data


def loadFolder(folderpath, dtype=float, **kwargs):
    # load all continuous files in a folder

    data = {}

    # load all continuous files in a folder
    if 'channels' in kwargs.keys():
        filelist = ['100_CH' + x + '.continuous' for x in map(str, kwargs['channels'])]
    else:
        filelist = os.listdir(folderpath)

    t0 = time.time()
    numFiles = 0

    for i, f in enumerate(filelist):
        if '.continuous' in f:
            data[f.replace('.continuous', '')] = loadContinuous(os.path.join(folderpath, f), dtype=dtype)
            numFiles += 1

    print(''.join(('Avg. Load Time: ', str((time.time() - t0) / numFiles), ' sec')))
    print(''.join(('Total Load Time: ', str((time.time() - t0)), ' sec')))

    return data


def loadFolderToArray(folderpath, channels='all', chprefix='CH',
                      dtype=float, session='0', source='100'):
    '''Load continuous files in specified folder to a single numpy array. By default all
    CH continous files are loaded in numerical order, ordering can be specified with
    optional channels argument which should be a list of channel numbers.'''

    if channels == 'all':
        channels = _get_sorted_channels(folderpath, chprefix, session, source)

    if session == '0':
        filelist = [source + '_' + chprefix + x + '.continuous' for x in map(str, channels)]
    else:
        filelist = [source + '_' + chprefix + x + '_' + session + '.continuous' for x in map(str, channels)]

    t0 = time.time()
    numFiles = 1

    channel_1_data = loadContinuous(os.path.join(folderpath, filelist[0]), dtype)['data']

    n_samples = len(channel_1_data)
    n_channels = len(filelist)

    data_array = np.zeros([n_samples, n_channels], dtype)
    data_array[:, 0] = channel_1_data

    for i, f in enumerate(filelist[1:]):
        data_array[:, i + 1] = loadContinuous(os.path.join(folderpath, f), dtype)['data']
        numFiles += 1

    print(''.join(('Avg. Load Time: ', str((time.time() - t0) / numFiles), ' sec')))
    print(''.join(('Total Load Time: ', str((time.time() - t0)), ' sec')))

    return data_array


def loadContinuous(filepath, dtype=float):
    assert dtype in (float, np.int16), \
        'Invalid data type specified for loadContinous, valid types are float and np.int16'

    print("Loading continuous data...")

    ch = {}

    # read in the data
    f = open(filepath, 'rb')

    fileLength = os.fstat(f.fileno()).st_size

    # calculate number of samples
    recordBytes = fileLength - NUM_HEADER_BYTES
    if recordBytes % RECORD_SIZE != 0:
        raise Exception("File size is not consistent with a continuous file: may be corrupt")
    nrec = recordBytes // RECORD_SIZE
    nsamp = nrec * SAMPLES_PER_RECORD
    # pre-allocate samples
    samples = np.zeros(nsamp, dtype)
    timestamps = np.zeros(nrec)
    recordingNumbers = np.zeros(nrec)
    indices = np.arange(0, nsamp + 1, SAMPLES_PER_RECORD, np.dtype(np.int64))

    header = readHeader(f)

    recIndices = np.arange(0, nrec)

    for recordNumber in recIndices:

        timestamps[recordNumber] = np.fromfile(f, np.dtype('<i8'), 1)  # little-endian 64-bit signed integer
        N = np.fromfile(f, np.dtype('<u2'), 1)[0]  # little-endian 16-bit unsigned integer

        # print index

        if N != SAMPLES_PER_RECORD:
            raise Exception('Found corrupted record in block ' + str(recordNumber))

        recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))  # big-endian 16-bit unsigned integer

        if dtype == float:  # Convert data to float array and convert bits to voltage.
            data = np.fromfile(f, np.dtype('>i2'), N) * float(
                header['bitVolts'])  # big-endian 16-bit signed integer, multiplied by bitVolts
        else:  # Keep data in signed 16 bit integer format.
            data = np.fromfile(f, np.dtype('>i2'), N)  # big-endian 16-bit signed integer
        samples[indices[recordNumber]:indices[recordNumber + 1]] = data

        marker = f.read(10)  # dump

    # print recordNumber
    # print index

    ch['header'] = header
    ch['timestamps'] = timestamps
    ch['data'] = samples  # OR use downsample(samples,1), to save space
    ch['recordingNumber'] = recordingNumbers
    f.close()
    return ch


def loadSpikes(filepath):
    '''
    Loads spike waveforms and timestamps from filepath (should be .spikes file)
    '''

    data = {}

    print('loading spikes...')

    f = open(filepath, 'rb')
    header = readHeader(f)

    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .spikes files with version 0.4 or higher')

    data['header'] = header
    numChannels = int(header['num_channels'])
    numSamples = 40  # **NOT CURRENTLY WRITTEN TO HEADER**

    spikes = np.zeros((MAX_NUMBER_OF_SPIKES, numSamples, numChannels))
    timestamps = np.zeros(MAX_NUMBER_OF_SPIKES)
    source = np.zeros(MAX_NUMBER_OF_SPIKES)
    gain = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    thresh = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    sortedId = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    recNum = np.zeros(MAX_NUMBER_OF_SPIKES)

    currentSpike = 0

    while f.tell() < os.fstat(f.fileno()).st_size:
        eventType = np.fromfile(f, np.dtype('<u1'), 1)  # always equal to 4, discard
        timestamps[currentSpike] = np.fromfile(f, np.dtype('<i8'), 1)
        software_timestamp = np.fromfile(f, np.dtype('<i8'), 1)
        source[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        numChannels = np.fromfile(f, np.dtype('<u2'), 1)[0]
        numSamples = np.fromfile(f, np.dtype('<u2'), 1)[0]
        sortedId[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        electrodeId = np.fromfile(f, np.dtype('<u2'), 1)
        channel = np.fromfile(f, np.dtype('<u2'), 1)
        color = np.fromfile(f, np.dtype('<u1'), 3)
        pcProj = np.fromfile(f, np.float32, 2)
        sampleFreq = np.fromfile(f, np.dtype('<u2'), 1)

        waveforms = np.fromfile(f, np.dtype('<u2'), numChannels * numSamples)
        gain[currentSpike, :] = np.fromfile(f, np.float32, numChannels)
        thresh[currentSpike, :] = np.fromfile(f, np.dtype('<u2'), numChannels)
        recNum[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)

        waveforms_reshaped = np.reshape(waveforms, (numChannels, numSamples))
        waveforms_reshaped = waveforms_reshaped.astype(float)
        waveforms_uv = waveforms_reshaped

        for ch in range(numChannels):
            waveforms_uv[ch, :] -= 32768
            waveforms_uv[ch, :] /= gain[currentSpike, ch] * 1000

        spikes[currentSpike] = waveforms_uv.T

        currentSpike += 1

    data['spikes'] = spikes[:currentSpike, :, :]
    data['timestamps'] = timestamps[:currentSpike]
    data['source'] = source[:currentSpike]
    data['gain'] = gain[:currentSpike, :]
    data['thresh'] = thresh[:currentSpike, :]
    data['recordingNumber'] = recNum[:currentSpike]
    data['sortedId'] = sortedId[:currentSpike]

    return data


def loadEvents(filepath):
    data = {}

    print('loading events...')

    f = open(filepath, 'rb')
    header = readHeader(f)

    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

    data['header'] = header

    index = -1

    channel = np.zeros(MAX_NUMBER_OF_EVENTS)
    timestamps = np.zeros(MAX_NUMBER_OF_EVENTS)
    sampleNum = np.zeros(MAX_NUMBER_OF_EVENTS)
    nodeId = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventType = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventId = np.zeros(MAX_NUMBER_OF_EVENTS)
    recordingNumber = np.zeros(MAX_NUMBER_OF_EVENTS)

    while f.tell() < os.fstat(f.fileno()).st_size:
        index += 1

        timestamps[index] = np.fromfile(f, np.dtype('<i8'), 1)
        sampleNum[index] = np.fromfile(f, np.dtype('<i2'), 1)
        eventType[index] = np.fromfile(f, np.dtype('<u1'), 1)
        nodeId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        eventId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        channel[index] = np.fromfile(f, np.dtype('<u1'), 1)
        recordingNumber[index] = np.fromfile(f, np.dtype('<u2'), 1)

    data['channel'] = channel[:index]
    data['timestamps'] = timestamps[:index]
    data['eventType'] = eventType[:index]
    data['nodeId'] = nodeId[:index]
    data['eventId'] = eventId[:index]
    data['recordingNumber'] = recordingNumber[:index]
    data['sampleNum'] = sampleNum[:index]

    return data


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header


def downsample(trace, down):
    downsampled = scipy.signal.resample(trace, np.shape(trace)[0] / down)
    return downsampled


def pack(folderpath, source='100', **kwargs):
    # convert single channel open ephys channels to a .dat file for compatibility with the KlustaSuite, Neuroscope and Klusters
    # should not be necessary for versions of open ephys which write data into HDF5 format.
    # loads .continuous files in the specified folder and saves a .DAT in that folder
    # optional arguments:
    #   source: string name of the source that openephys uses as the prefix. is usually 100, if the headstage is the first source added, but can specify something different
    #
    #   data: pre-loaded data to be packed into a .DAT
    #   dref: int specifying a channel # to use as a digital reference. is subtracted from all channels.
    #   order: the order in which the .continuos files are packed into the .DAT. should be a list of .continious channel numbers. length must equal total channels.
    #   suffix: appended to .DAT filename, which is openephys.DAT if no suffix provided.

    # load the openephys data into memory
    if 'data' not in kwargs.keys():
        if 'channels' not in kwargs.keys():
            data = loadFolder(folderpath, dtype=np.int16)
        else:
            data = loadFolder(folderpath, dtype=np.int16, channels=kwargs['channels'])
    else:
        data = kwargs['data']
    # if specified, do the digital referencing
    if 'dref' in kwargs.keys():
        ref = load(os.path.join(folderpath, ''.join((source, '_CH', str(kwargs['dref']), '.continuous'))))
        for i, channel in enumerate(data.keys()):
            data[channel]['data'] = data[channel]['data'] - ref['data']
    # specify the order the channels are written in
    if 'order' in kwargs.keys():
        order = kwargs['order']
    else:
        order = list(data)
    # add a suffix, if one was specified
    if 'suffix' in kwargs.keys():
        suffix = kwargs['suffix']
    else:
        suffix = ''

    # make a file to write the data back out into .dat format
    outpath = os.path.join(folderpath, ''.join(('openephys', suffix, '.dat')))
    out = open(outpath, 'wb')

    # go through the data and write it out in the .dat format
    # .dat format specified here: http://neuroscope.sourceforge.net/UserManual/data-files.html
    channelOrder = []
    print(''.join(('...saving .dat to ', outpath, '...')))
    random_datakey = next(iter(data))
    bar = ProgressBar(len(data[random_datakey]['data']))
    for i in range(len(data[random_datakey]['data'])):
        for j in range(len(order)):
            if source in random_datakey:
                ch = data[order[j]]['data']
            else:
                ch = data[''.join(('CH', str(order[j]).replace('CH', '')))]['data']
            out.write(struct.pack('h', ch[i]))  # signed 16-bit integer
            # figure out which order this thing packed the channels in. only do this once.
            if i == 0:
                channelOrder.append(order[j])
        # update how mucb we have list
        if i % (len(data[random_datakey]['data']) / 100) == 0:
            bar.animate(i)
    out.close()
    print(''.join(('order: ', str(channelOrder))))
    print(''.join(('.dat saved to ', outpath)))


# **********************************************************
# progress bar class used to show progress of pack()
# stolen from some post on stack overflow
import sys

try:
    from IPython.display import clear_output

    have_ipython = True
except ImportError:
    have_ipython = False


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r', self, )
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


# *************************************************************

def pack_2(folderpath, filename='', channels='all', chprefix='CH',
           dref=None, session='0', source='100'):
    '''Alternative version of pack which uses numpy's tofile function to write data.
    pack_2 is much faster than pack and avoids quantization noise incurred in pack due
    to conversion of data to float voltages during loadContinous followed by rounding
    back to integers for packing.
    filename: Name of the output file. By default, it follows the same layout of continuous files,
              but without the channel number, for example, '100_CHs_3.dat' or '100_ADCs.dat'.
    channels:  List of channel numbers specifying order in which channels are packed. By default
               all CH continous files are packed in numerical order.
    chprefix:  String name that defines if channels from headstage, auxiliary or ADC inputs
               will be loaded.
    dref:  Digital referencing - either supply a channel number or 'ave' to reference to the
           average of packed channels.
    source: String name of the source that openephys uses as the prefix. It is usually 100,
            if the headstage is the first source added, but can specify something different.
    '''

    data_array = loadFolderToArray(folderpath, channels, chprefix, np.int16, session, source)

    if dref:
        if dref == 'ave':
            print('Digital referencing to average of all channels.')
            reference = np.mean(data_array, 1)
        else:
            print('Digital referencing to channel ' + str(dref))
            if channels == 'all':
                channels = _get_sorted_channels(folderpath, chprefix, session, source)
            reference = deepcopy(data_array[:, channels.index(dref)])
        for i in range(data_array.shape[1]):
            data_array[:, i] = data_array[:, i] - reference

    if session == '0':
        session = ''
    else:
        session = '_' + session

    if not filename: filename = source + '_' + chprefix + 's' + session + '.dat'
    print('Packing data to file: ' + filename)
    data_array.tofile(os.path.join(folderpath, filename))


def _get_sorted_channels(folderpath, chprefix='CH', session='0', source='100'):
    Files = [f for f in os.listdir(folderpath) if '.continuous' in f
             and '_' + chprefix in f
             and source in f]

    if session == '0':
        Files = [f for f in Files if len(f.split('_')) == 2]
        Chs = sorted([int(f.split('_' + chprefix)[1].split('.')[0]) for f in Files])
    else:
        Files = [f for f in Files if len(f.split('_')) == 3
                 and f.split('.')[0].split('_')[2] == session]

        Chs = sorted([int(f.split('_' + chprefix)[1].split('_')[0]) for f in Files])

    return (Chs)


# This is where I start to mess around
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.fft
import random
from scipy import signal
from scipy.integrate import simps
from math import floor

# Constants
fs = 1000
NPERSEG = 1000
EPOCH_LENGTH = 4
FREQ_BINS = 501
FACTOR = 30

DELTA_LOW = 1
DELTA_HIGH = 4

THETA_LOW = 6
THETA_HIGH = 10

GAMMA_LOW = 55
GAMMA_HIGH = 70

STD_ALLOWED = 2

# plots a PSD
def plot_PSD(signal, fs):
    """
    Plots a PSD of a signal with sampling frequency fs
    """
    f, Pxx_den = scipy.signal.welch(signal, fs=fs)
    return plt.semilogy(f, Pxx_den)

def save_PSD(signal, title, xlab='frequency [Hz]', ylab='PSD [V**2/Hz]'):
    """"
    Plots PSD in 0-50Hz range
    """
    f, Pxx_den = scipy.signal.welch(signal, fs=fs)
    plt.semilogy(f[0:22], Pxx_den[0:22])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(fname=title)
    plt.close()

def save_plot(signal, title, xlab="Time (ms)", ylab="Amplitude (mV)"):
    x = (range(0, len(signal)))

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.plot(x, signal)
    plt.savefig(fname=title)
    plt.close()

def plot_scatter(x, y, labels, title, xlabel, ylabel):
    """"
    Plots a scatterplot given labels and x and y.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, c=labels, label=labels, s=.75)

def filter_design(order, fs):
    """
    Creates a filter given an order and sampling frequency, returns the filter in sos
    """
    sos = signal.iirfilter(order, [48, 52], btype='bandstop', analog=False, ftype='butter', fs=fs, output='sos')
    return sos


def epoch_split(signal, epoch_length, fs):
    """
    Splits signal into epochs of specified length.
    Returns the split signal in a numpy array with shape (n_epochs, length-epochs).
    """
    signal_len = len(signal)
    epoch_n = floor(signal_len / (epoch_length * fs))
    signal_split = signal[0:int(epoch_n * fs * epoch_length)]
    epochs = np.split(signal_split, epoch_n)

    print('Signal split into epochs')
    return epochs


def power_calc(psd, freqs, band_low, band_high):
    """
    Calculates band power given edges of bands and power spectrum of an epoch.
    """
    idx_band = np.logical_and(freqs >= band_low, freqs <= band_high)

    freq_res = freqs[1] - freqs[0]
    band_power = simps(psd[idx_band], dx=freq_res)

    return band_power


# bands might apparently be an issue
def epoch_welch(epochs, fs, nperseg, bands=[(DELTA_LOW, DELTA_HIGH), (THETA_LOW, THETA_HIGH)]):
    """
    Performs a power spectrum density analysis on epochs given the sampling frequency
    and length of segment to use for the Welch's method.
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


def resampling(signal, factor=FACTOR):
    size = len(signal)

    print('Starting resampling...')
    data_resampled = scipy.signal.resample(signal, round(size / factor))

    return data_resampled

def analyse(filepath, loaded=False, resampled=False):
    """"
    Performs all analysis needed given a filepath.
    loaded indicates whether filepath is a string or an array of points.
    """
    # loads data or continues processing
    if not loaded:
        data = load(filepath)
        data = data['data']
    else:
        data = filepath

    # downsamples data
    if not resampled:
        try:
            data_downsampled = resampling(data)
        except TypeError:
            raise Exception('Check what gets passed to the analyse function')
    else:
        data_downsampled = data

    # creates and applies filter
    sos = filter_design(17, fs=fs)
    data_filter = scipy.signal.sosfilt(sos, data_downsampled)
    print('Filter applied')

    # splits into epochs of length EPOCH_LENGTH
    epochs = epoch_split(data_filter, EPOCH_LENGTH, fs=fs)

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

def average_tetrode(signals):
    n_channels = len(signals[0, :])
    tetrode_average = (np.sum(signals, axis=1) / n_channels)

    return tetrode_average


def filepath_creation(n_channel, type=1):
    if type == 1:
        file_extension = "100_CH" + str(n_channel)
    elif type == 2:
        file_extension = "100_" + str(n_channel + 20)
    else:
        file_extension = "100_CH" + str(n_channel) + "_2"

    file_extension = file_extension + ".continuous"
    return file_extension

# Prepare script for cluster
main_path = 'data/jpatriota/'

path_day1 = ["1.Habituation/Habituation/2020-11-13_01-23-17/Record Node 117",
             "1.Habituation/Habituation/2020-11-13_01-23-17/Record Node 125",
             "1.Habituation/Habituation/r12 sleep_hab/rec raw/2020-11-13_04-49-07/Record Node 117"]

# convention: 100_CH#.cont
path_day2 = ["2.Fear conditioning/Fear conditioning/fc_sleep/raw/2020-11-14_02-09-18/Record Node 117",
            "2.Fear conditioning/Fear conditioning/Fear conditioning awake/raw/2020-11-14_00-59-06/Record Node 117"]

# convention: 100_#.continuous
path_day3 = ["3.Probe test/probe test/Probe fear/probe awake/raw/2020-11-15_00-38-02/Record Node 117",
             "3.Probe test/probe test/Probe slee/raw/2020-11-15_02-16-08/Record Node 117"]

# convention: 100_#.continuous
path_day4 = ["4.Extinction training/Extinction learning/Extinction learning awake/raw/2020-11-16_00-33-35/Record Node 117",
            "4.Extinction training/Extinction learning/Extinction learning sleep/raw/2020-11-16_03-21-15/Record Node 117"]

# convention: 100_TYPE#.continuous
path_day5 = ["5.Extinction test/extinction probe awake/raw/2020-11-17_00-44-46/Record Node 117",
            "5.Extinction test/extinction probe sleep/raw/2020-11-17_02-01-55/Record Node 117"]

path_day1 = list(path_day1.pop(-1))
# makes list of all data paths
data_paths = [path_day1, path_day2, path_day3, path_day4, path_day5]
ch_channels = list(range(21, 129))

# iteration variable
count = 2

# preps list for later saving
power_spectrums = list()

# checking paths for running on the cluster
base_path = os.path.abspath(os.curdir)
print(base_path)
os.chdir("..")
os.chdir("..")
root_path = os.path.abspath(os.curdir)
print(root_path)

# loop over every recording day
for days in data_paths:
    # loop over every recording of interest
    for recording in days:
        # set variable for correct filepath creation
        if count in [0, 3, 4, 9, 10]:
            type = 1
        elif count in [1, 5, 6, 7, 8]:
            type = 2
        else:
            type = 3

        # iterate counting for proper loading of channels
        count = count + 1

        if recording == str(1):
            root_filepath = main_path + "1.Habituation/Habituation/r12 sleep_hab/rec raw/2020-11-13_04-49-07/Record Node 117"
        else:
            root_filepath = main_path + recording
        powers = np.array([])

        # loops over all tetrodes
        for i in range(0, 32): #(0,32)
            # loops over each channel in tetrode
            for j in range(1, 5): # for j in range(1,5):
                # gets exact channel number
                n_channel = i * 4 + j

                # creates full filepath
                extension = filepath_creation(n_channel=n_channel, type=type)
                filepath = root_filepath + '/' + extension

                # loads in signal
                data = load(filepath)
                LFP_signal = data['data']

                print(f"completed loading of {filepath}")

                # resamples channel
                LFP_data = resampling(LFP_signal)

                # Checks for the existence of a plots folder
                plot_filepath = root_filepath + '/' + 'plots'
                if not os.path.isdir(plot_filepath):
                    os.makedirs(plot_filepath)

                # saves a power spectrum and basic plot in specified location
                os.chdir(plot_filepath)
                file_title = extension.replace('.continuous', '') + '_PSD'
                save_PSD(LFP_signal, title=file_title)

                signal_len = len(LFP_signal)
                plot_start = random.randint(0, signal_len - 100 * fs)

                file_title = extension.replace('.continuous', '') + '_plot'
                save_plot(signal=LFP_signal[plot_start:plot_start + 15 * fs], title=file_title)
                # changes working directory back to root
                os.chdir(root_path)

""""
                # perform power analysis of theta and delta for each channel
                power = analyse(LFP_signal, loaded=True)

                # data comes out of analyse in shape (n_epochs, n_band)
                if powers.size > 0:
                    powers = np.append(powers, power, axis=1)
                else:
                    powers = power

"""


""""
                # adds signals together
                if LFP_data.size > 0:
                    # trims signal if signals don't match within a tetrode
                    try:
                        LFP_data = LFP_data + LFP_signal
                    except ValueError:
                        if LFP_data.shape[0] > LFP_signal.shape[0]:
                            LFP_data = LFP_data[:LFP_signal.shape[0], :]
                            LFP_data = LFP_data + LFP_signal
                        else:
                            LFP_signal = LFP_signal[:LFP_data.shape[0], :]
                            LFP_data = LFP_data + LFP_signal
                else:
                    LFP_data = LFP_signal
"""

#        power_spectrums.append(powers)

os.chdir(base_path)

# save results to pickle file
with open('LFP_power.pickle', 'wb') as f:
    pickle.dump(power_spectrums, f)
