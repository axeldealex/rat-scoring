# package imports
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import scipy
from sklearn.mixture import GaussianMixture
import pandas as pd

# handmade function imports
from LFP_util import resampling, epoch_welch, power_calc
from epoch_util import determine_epochs, determine_epoch_timestamps, epoch_split
from path_util import filepath_creation
from movement_util import movement_preprocess, calculateDistance, smoothTriangle, analyze_dlc_data

# constants - own
TTL_resampled_fs = 1000
STEP_LENGTH = 0.1
FACTOR = 30
MAX_SHIFT = 10
LEN_EPOCHS = 4
N_EPOCHS = 20
NPERSEG = 1000
N_BANDS = 2
TETRODE_ELECTRODES = 4

N_CHANNELS = 128
N_TETRODES = int(N_CHANNELS / TETRODE_ELECTRODES)

# constants - openephys
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
    else:
        raise Exception("Not a recognized file type. Please input a .continuous, .spikes, or .events file")

    return data


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header


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


def downsample_TTL(ttl_signal):
    """"
    downsamples TTL signal from 30kHz to 1000Hz
    """
    ds_array = []
    time_axis = []
    for k in range(0, len(ttl_signal), 30):
        ds_array.append(ttl_signal[k])
        #TODO
        # commented this out for now, should reinstate it and fix function
        # time_axis.append(time[k])

    return ds_array, time_axis


def TTL_analysis(filepath):
    # loads in TTL channel
    ttl_channel = load(filepath)
    ttl_channel = ttl_channel['data']

    # checks some variables
    size = len(ttl_channel)
    maximum = np.amax(ttl_channel)
    fs = 30000

    t = np.array(range(0, (size)))  # making the time axis
    time = t / fs  # getting time axis in seconds

    # down sampling from 30khz to 1000hz (jumps of 30)
    ds_array, time_axis = downsample_TTL(ttl_channel)

    # initialise variables for looping
    fs_new = 1000
    n_elements_ds = (len(ds_array))
    i = 0
    prev = False
    ttl_final = np.zeros(len(ds_array))
    time_axis2 = np.array(time_axis)

    # set boundaries for on/off detection
    ttl_on_upper = maximum * 101
    ttl_on_lower = maximum * 98

    # sets step size
    step = int(fs_new * STEP_LENGTH)

    # marks start of TTL signals
    # loops through whole signal
    while i < n_elements_ds - 1:
        # if ttl value is close to max
        if ds_array[i] > 0.98 * maximum:
            # sum consecutive points
            sum_sec = sum(ds_array[i: i + step])
            # within boundary of on
            if ttl_on_lower < sum_sec <= ttl_on_upper:
                # sets value at 5 and increments
                ttl_final[i:i+step] = 5
                i = i + step

        i = i + 1

    return ttl_final


def setboundaries(lum_array):
    """"
    Calculates boundaries of on/off given an array of luminance values
    """
    maximum = max(lum_array)
    bound_high = int(maximum) + 1
    bound_low = round(maximum * 0.75)

    return bound_low, bound_high


def extract_led_status(lum_array, n_elements, bound_high, bound_low):
    i = 0
    LED_on = 0
    led_vid = np.zeros(n_elements)
    while i < n_elements - 1:
        if bound_high > lum_array[i] > bound_low:
            led_vid[i] = 5
            LED_on = LED_on + 1
        i = i + 1

    return led_vid, LED_on


def correlation_calc(series1, series2, fs):
    """"
    Calcultaes overlap between two series: assumes a binary signal for both series 1 and 2.
    Returns overlap_score, an arbitrary unit of likeness.
    """
    overlap_score = np.multiply(series1, series2)

    return overlap_score


def make_binary(signal, threshold):
    """"
    Makes signal binary, making each value 0 or 1 based on a given threshold value.
    Returns the binary signal.
    """
    # trims signal below threshold
    low = np.where(signal < threshold)[0]
    signal[low] = 0

    # trims signal above threshold
    high = np.where(signal > threshold)[0]
    signal[high] = 1
    return signal


def check_overlap(sample, main, start_index, max_shift, fs):
    """"
    Given a sample and a comparison start index, slides sample signal over main signal and returns an array containing
    overlap at specific shifts.
    """
    # initialises borders to check
    check_border = int(max_shift * fs)
    left_compare = start_index - check_border
    right_compare = start_index + check_border

    # sets hard limits to borders
    if left_compare < 0:
        left_compare = 0
    elif right_compare > len(main):
        right_compare = len(main)

    # initialises vars for loop
    len_sample = len(sample)
    overlap_calc = []
    curr = 0
    best_match = -1

    # performs sliding comparison of signals within range
    for i in range(left_compare, right_compare, 1):
        # gets part of main signal to make comparison to
        main_compare = main[i:i+len_sample]

        # calculates overlap between sample and part of main
        overlap = sum(correlation_calc(sample, main_compare, video_fs))
        overlap_calc.append(overlap)

        # if calculated overlap is higher than prev high, replaces and notes index
        if overlap > curr:
            curr = overlap
            best_match = i

    return overlap_calc, curr, best_match


def plot_realignment_check(ttl_signal, led_signal, video_timestamps, len_plot=500):
    """"
    Goes over ttl  and led signal and creates and saves plots to show realignment.
    WARNING: this creates a large amount of plots: I don't recommend using this unless you're going over your data
    for the first time and want a visual confirmation.
    """
    plt.close()
    for k in range(0, int(len(ttl_signal) / 500) - 1):
        start_plot = len_plot * k

        plt.plot(video_timestamps[start_plot:start_plot + len_plot], ttl_signal[start_plot:start_plot + len_plot])
        plt.plot(video_timestamps[start_plot:start_plot + len_plot], led_signal[start_plot:start_plot + len_plot])
        plt.savefig(fname=str(k))
        plt.close()


# changes main working directory to root
os.chdir("..")
os.chdir("..")
root_path = os.path.abspath(os.curdir)
print(root_path)

########################################################################
# LED video analysis
#
# Analyses video for LED luminance, later used to align video with TTL. Some dropped frames, but general functionality
# is good. Code mostly taken from StackOverflow.

# Video has to be of isolated LED: I personally used Movavi, although others also seem reasonable.
# Quality of video doesn't have to be great, so use low resolution and file size (saves on rendering time)
########################################################################
os.chdir(r"C:\Users\Alex\PycharmProjects\pythonProject")
# Create a VideoCapture object and read from input file
video_path = "extinction_awaker120LED.avi"

cap = cv2.VideoCapture(video_path)
# Check if camera opened successfully
if not cap.isOpened():
  print("Error opening video stream or file")

# get number of frames in video
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# initialises array to save values
luminance_array = [0] * int(n_frames)
n_frame = 0

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if n_frame % 5000 == 0:
        print(f"currently on frame {n_frame} out of {n_frames}")
    # if there is another frame
    if ret:
        # calculates luminance for the frame and saves it
        dims = frame.size
        luminance_frame = np.sum(frame) / dims
        luminance_array[n_frame] = luminance_frame
        n_frame = n_frame + 1
    # or break the loop
    else:
        break

print("Extracted luminance from video")

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# sets boundaries of on/off
bound_low, bound_high = setboundaries(luminance_array)

# intializes variables for loop
n_elements_vid = len(luminance_array)

# goes though luminance array and extracts when TTL turns on
led_vid, LED_on = extract_led_status(luminance_array, n_elements_vid, bound_high, bound_low)
print("Finished extracting LED data")

# TTL data is 1000Hz after resampling
ttl_path = "100_3.continuous"

print("Started digital TTL loading")
ttl_digital = TTL_analysis(ttl_path)

# checks how often the signal is on/off
flashes_digital = len(np.where(ttl_digital == 5)[0])
flashes_LED = len(np.where(led_vid == 5)[0])
print(f"There are {flashes_digital} on signals digitally and {flashes_LED} physically")

# defines length of recording and video sample rate
time_recording = len(ttl_digital) / 1000
video_fs = n_frames / time_recording

# makes video time axis, assuming constant framerate
video_timestamps = np.linspace(0, n_frames * (1 / video_fs), int(n_frames))

# makes TTL time axis
ttl_timestamps = np.linspace(0, len(ttl_digital) / TTL_resampled_fs, len(ttl_digital))

# Checks if endpoints of timestamps match
if round(ttl_timestamps[-1], 3) == round(video_timestamps[-1], 3):
    print('Endpoints of timestamps match')
else:
    print("Endpoints of timestamps do not match")

# resamples the digital signal to get same fs
n_resampled = round(len(ttl_digital) / (TTL_resampled_fs / video_fs))
ttl_digital_resampled = scipy.signal.resample(ttl_digital, n_resampled)

# levels out both signal to be completely binary
ttl_digital_resampled = make_binary(ttl_digital_resampled, 3)
led_vid_resampled = make_binary(led_vid, 3)


########################################################################
# Realignment
# Realigns TTL and video signal: based on LED luminance and TTL on/off state, epochs are defined and timestamps
# extracted.
########################################################################

# find on-state in the digital signal
dig_peaks = np.where(ttl_digital_resampled == 1)[0]

# identifies start of on-state
dig_peaks_start = []
for j in range(len(dig_peaks)):
    if dig_peaks[j] - 1 != dig_peaks[j - 1]:
        dig_peaks_start.append(dig_peaks[j])

# gets index of peaks 6-10 of digital signal to use in alignment
dig_sample_start = dig_peaks_start[5]
# grabs end of off-state before 11th peak
dig_sample_end = dig_peaks_start[10] - 1

# gets sample signal to compare
dig_sample = ttl_digital_resampled[dig_sample_start:dig_sample_end]
len_sample = len(dig_sample)

# sets start and end of comparison window
start_compare = dig_sample_start - int(MAX_SHIFT * video_fs)
end_compare = dig_sample_start + int(MAX_SHIFT * video_fs)

if start_compare < 0:
    start_compare = 0

# compares sample digital signal to multiple samples of the LED signal
overlap_calc, max_overlap, best_match_index = check_overlap(dig_sample, led_vid_resampled, dig_sample_start,
                                                            max_shift=MAX_SHIFT, fs=video_fs)

# calculates shift of good fit based on highest overlap
shift = best_match_index - dig_sample_start

# resamples signals from start of digital sample and point of best fit
aligned_ttl = ttl_digital_resampled[dig_sample_start:]
aligned_ttl_timestamps = video_timestamps[dig_sample_start:]

# realigns the time axis of both signals
aligned_led = led_vid_resampled[best_match_index:]
aligned_led_timestamps = video_timestamps[best_match_index:]
first_aligned_frame = best_match_index

# shortens end of longer signal
diff = len(aligned_ttl) - len(aligned_led)
if diff > 0:
    aligned_ttl = np.delete(aligned_ttl, np.linspace(-diff, -1, diff, dtype=int))
elif diff < 0:
    aligned_led = np.delete(aligned_led, np.linspace(diff, -1, -diff, dtype=int))

# calculates alignment of the two realigned signals
total_overlap = correlation_calc(aligned_ttl, aligned_led, video_fs)
print(f"Total overlap is {sum(total_overlap)} out of expected {flashes_LED} for {sum(total_overlap)/flashes_LED * 100}%")

# determine values for 20 epochs
epoch_frames = determine_epochs(first_aligned_frame, video_fs)
epoch_timestamps = determine_epoch_timestamps(first_aligned_frame, epoch_frames, aligned_ttl_timestamps)

# initialises variables for loop
max_deviation = 0.0
max_comparisons = round(len(aligned_ttl) / (epoch_frames[-1][1] - epoch_frames[0][0]))
running = True

epoch_frames_all = epoch_frames
epoch_timestamps_all = epoch_timestamps

# starts loop of realignment and epoch definition for entire signal
while running:
    max_deviation = max_deviation + 0.1
    # prepares comparison signal to perform alignment
    # based on end of last properly aligned epoch
    start_unaligned = epoch_frames[-1][1] - first_aligned_frame
    end_unaligned = start_unaligned + len(dig_sample)
    # grabs comparison signal from ttl data
    dig_sample_loop = aligned_ttl[start_unaligned:end_unaligned]

    # calculates overlap and returns point of highest overlap
    overlap_calc, max_overlap, best_match_index_loop = check_overlap(dig_sample_loop, aligned_led, start_unaligned,
                                                                     max_shift=max_deviation, fs=video_fs)

    print(f"Best match was found at frame {best_match_index_loop}, while first alignment suggested {start_unaligned}")

    # determine values for 20 epochs
    epoch_frames = determine_epochs(best_match_index_loop, video_fs, length_epochs=LEN_EPOCHS, n_epochs=N_EPOCHS)
    epoch_timestamps = determine_epoch_timestamps(first_aligned_frame, epoch_frames, aligned_ttl_timestamps)

    # saves timestamps and frames in variables
    epoch_frames_all.extend(epoch_frames)
    epoch_timestamps_all.extend(epoch_timestamps)

    # checks if loop should stop running based on remaining length of signal
    if video_timestamps[-1] - (epoch_timestamps[-1][1] + (N_EPOCHS * LEN_EPOCHS)) < 0:
        running = False

########################################################################
# LFP analysis
# Goes over all channels within a recording day and analyses the LFP signal for power within the Delta (1-4Hz) and
# Theta bands (6-10Hz). Also saves LFP power per tetrode and per area for later use in case of crashing.
########################################################################

# starts LFP processing
LFP_path = "/data/jpatriota/4.Extinction training/Extinction learning/Extinction learning awake/raw/2020-11-16_00-33-35/Record Node 117/"

# sets variables needed for loop
indexes_flag = False
timestamps_flag = False

# prepares array for saving powers per channel
shape_power_sum = (len(epoch_timestamps_all), N_BANDS, TETRODE_ELECTRODES)
power_per_band_sum = np.zeros(shape_power_sum)

# prepares array for saving power per tetrode
shape_power_tetrode = (len(epoch_timestamps_all), N_BANDS, N_TETRODES)
power_per_band_tetrode = np.zeros(shape_power_tetrode)

# loops over all tetrodes
for i in range(0, 32):
    # and all channels in tetrode
    for j in range(1, 5):
        # sets channel number
        n_channel = i * 4 + j

        # creates full filepath
        extension = filepath_creation(n_channel, type=2)
        channel_filepath = LFP_path + extension

        # loads in signal
        data = load(channel_filepath)
        LFP_signal = data['data']

        print(f"completed loading of {channel_filepath}")

        # resamples channel
        LFP_resampled = resampling(LFP_signal, factor=FACTOR)

        # creates timestamps for LFP channel if not done already
        if not timestamps_flag:
            LFP_timestamps = np.linspace(0, len(LFP_signal) / 1000, len(LFP_signal) + 1)
            timestamps_flag = True

            # need to round to 3 decimals for epoch_split to function correctly
            for k in range(0, len(LFP_timestamps)):
                LFP_timestamps[k] = round(LFP_timestamps[k], 3)

        # starts analysis of LFP channel
        if not indexes_flag:
            # if indexes have not been found yet, find them
            epochs_split, epoch_indexes = epoch_split(LFP_resampled, LFP_timestamps, epoch_timestamps_all)
            indexes_flag = True
        else:
            # fast method if indexes have been found
            epochs_split, epoch_indexes = epoch_split(LFP_resampled, LFP_timestamps, epoch_timestamps_all,
                                                      epoch_indexes=epoch_indexes, indexes=True)

        # calculates powers per band per epoch
        freqs, powers, power_per_band = epoch_welch(epochs_split, fs=TTL_resampled_fs, nperseg=NPERSEG)

        # saves calculated powers
        power_per_band_sum[:, :, j-1] = power_per_band

    # averages powers for each channel within tetrode
    power_per_band_tetrode[:, :, i] = np.sum(power_per_band_sum, axis=2) / TETRODE_ELECTRODES
    # empties variable
    power_per_band_sum = np.zeros(shape_power_sum)


# changes working directory
os.chdir("/home/11648775/dataClustering/")

# save tetrode powers to pickle file
power_per_band_tetrode_save = np.log10(power_per_band_tetrode)

# dumps LFP power per tetrode for later checking:
with open('LFP_power_tetrodes_day4awake.pickle', 'wb') as f:
    pickle.dump(power_per_band_tetrode_save, f)

# dumps indexes of LFP signals per epoch
if indexes_flag:
    with open('epoch_timestamps_day4awake.pickle', 'wb') as f:
        pickle.dump(epoch_indexes)

del power_per_band_tetrode_save

# returns working directory to root
os.chdir("..")
os.chdir("..")
os.chdir("..")

# separates tetrodes into their areas and calculates average power per band
power_hc = power_per_band_tetrode[:, :, 0:7]
power_hc = np.sum(power_hc, axis=2) / power_hc.shape[2]

power_ac = power_per_band_tetrode[:, :, 7:16]
power_ac = np.sum(power_ac, axis=2) / power_ac.shape[2]

# appends powers per area together
fit_object_lfp = np.append(power_hc, power_ac, axis=1)

power_bla = power_per_band_tetrode[:, :, 16:24]
power_bla = np.sum(power_bla, axis=2) / power_bla.shape[2]
fit_object_lfp = np.append(fit_object_lfp, power_bla, axis=1)

power_mpfc = power_per_band_tetrode[:, :, 24:]
power_mpfc = np.sum(power_mpfc, axis=2) / power_mpfc.shape[2]

# finishes fit object and takes log10
fit_object_lfp = np.append(fit_object_lfp, power_mpfc, axis=1)
fit_object_lfp = np.log10(fit_object_lfp)

# changes working directory
os.chdir("/home/11648775/dataClustering/")

# save tetrode powers to pickle file
with open('LFP_power_areas_day4awake.pickle', 'wb') as f:
    pickle.dump(fit_object_lfp, f)

# returns working directory to root
os.chdir("..")
os.chdir("..")
os.chdir("..")

########################################################################
# DLC movement data analysis
# Goes over csv created by DeepLabCut to extract movement data: uses frame numbers defined in alignment to extract
# correct parts of the file. After data has been processed, saves fit object for later use in case of crashes.
#TODO:
# change triangle to better working alternative of drive/back combined measure
########################################################################

# extracts movement information
movement_filepath = "/data/jpatriota/4.Extinction training/Extinction learning/Extinction learning awake/extinction_awaker120DLC_resnet50_DriveNetwork2Dec15shuffle1_1030000.csv"
MIN_MOVEMENT = 2
movement_conf, movement_data = analyze_dlc_data(movement_filepath, type="csv")

# extracts movement data, triangle of drive left/right and back
back_data = movement_data[:, -1]

# pre allocates var for data storage
movement_data_processed = np.zeros((len(epoch_frames_all), 2))

# prepares for drive selection, #TODO this is hard coded
movement_data_raw = pd.read_csv(movement_filepath, low_memory=False)
headers = list(movement_data_raw.columns)[1:]

drive1x = np.array(movement_data_raw[headers[12]][2:].astype(float))
drive1y = np.array(movement_data_raw[headers[13]][2:].astype(float))
drive1conf = np.array(movement_data_raw[headers[14]][2:].astype(float))
drive2x = np.array(movement_data_raw[headers[15]][2:].astype(float))
drive2y = np.array(movement_data_raw[headers[16]][2:].astype(float))
drive2conf = np.array(movement_data_raw[headers[17]][2:].astype(float))

from movement_util import drive_selection
# Determines movement of drives
drive_movement = drive_selection(drive1x, drive1y, drive1conf, drive2x, drive2y, drive2conf)

# determines movement measures per epoch
for i in range(0, len(epoch_frames_all)):
    # determines start and end frame of epoch
    start_frame = epoch_frames_all[i][0]
    end_frame = epoch_frames_all[i][1]
    len_epoch = end_frame - start_frame

    # determines average movement for right drive within frames of epoch
    movement_data_processed[i, 0] = sum(back_data[start_frame: end_frame]) / len_epoch
    movement_data_processed[i, 1] = sum(drive_movement[start_frame: end_frame]) / len_epoch

# takes average of drive and back
movement_data_processed = (np.nansum(movement_data_processed, axis=1) / 2)

# appends to fit object
fit_object = np.append(fit_object_lfp, movement_data_processed, axis=1)

# saves fit object
# changes working directory
os.chdir("/home/11648775/dataClustering/")

# save tetrode powers to pickle file
with open('fit_object_movement_day4awake.pickle', 'wb') as f:
    pickle.dump(fit_object, f)

# returns working directory to root
os.chdir("..")
os.chdir("..")
os.chdir("..")