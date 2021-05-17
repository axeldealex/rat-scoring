import numpy as np


def determine_epochs(aligned_frame_index, fs, length_epochs=4, n_epochs=20):
    """"
    Determines start and end of n_epochs by frame number.
    Returns a list of tuples in format (first_frame_epoch, last_frame_epoch)
    """
    borders = []
    # loops over all epochs
    for i in range(0, n_epochs):
        start_epoch = round(aligned_frame_index + i * length_epochs * fs)
        end_epoch = round(aligned_frame_index + (i+1) * length_epochs * fs) - 1
        epoch_borders = (start_epoch, end_epoch)
        borders.append(epoch_borders)

    return borders


def determine_epoch_timestamps(first_aligned_frame, borders, ttl_timestamps, offset_frames=0):
    """"
    Given first aligned frame, frame numbers and timestamps, returns timestamps of epochs.
    """
    timestamps = []
    # goes over all epochs
    for i in range(0, len(borders)):
        # determines start and end of epoch by index
        start_epoch = borders[i][0] - first_aligned_frame
        end_epoch = borders[i][1] - first_aligned_frame

        # grabs exact timestamps based on index
        start_epoch_timestamp = round(ttl_timestamps[start_epoch], 3)
        end_epoch_timestamp = round(ttl_timestamps[end_epoch], 3)

        # records and saves start/end timestamp for later
        epoch_timestamps = (start_epoch_timestamp, end_epoch_timestamp)
        timestamps.append(epoch_timestamps)

    return timestamps


def epoch_split(signal, signal_timestamps, epoch_timestamps, epoch_indexes=0, indexes=False):
    """"
    Splits epochs according to given timestamps, returning a list of lists.
    List has shape (n_epochs, 1) with each entry having shape (1, len_epoch).
    """
    # defines number of epochs
    epoch_n = len(epoch_timestamps)
    epochs_split = [[] for i in range(epoch_n)]

    # checks if indexes have already been given: if not, allocates variable for recording them if not
    if not indexes:
        epoch_indexes = [[0, 0] for i in range(epoch_n)]

    # runs for every epoch
    for i in range(0, epoch_n):
        # communicates running to user
        if (i-1) % 500 == 0 and i != 0:
            print(f"{i} epochs analysed")

        # if indexes are given, runs through epoch selection faster
        if indexes:
            start_epoch = epoch_indexes[i][0]
            end_epoch = epoch_indexes[i][1]
        else:
            # otherwise finds matching timestamps and saves indexes for later faster running
            start_epoch = np.where(signal_timestamps == epoch_timestamps[i][0])[0][0]
            end_epoch = np.where(signal_timestamps == epoch_timestamps[i][1])[0][0]
            epoch_indexes[i] = [start_epoch, end_epoch]

        # isolates part of signal within timestamps and saves
        epoch_signal = signal[start_epoch:end_epoch]
        epochs_split[i] = epoch_signal

    return epochs_split, epoch_indexes
