import numpy as np
import math
import pandas as pd

MIN_MOVEMENT = 2
EPOCH_LENGTH = 120

def movement_preprocess(data, confidence, p_cutoff=0.60):
    """"
    Preprocesses movement data by smoothing across intervals of low confidence.
    """
    bad_data = False
    bad_index = []

    #TODO fix this bug
    # this is not good, will cause very bad data in samples where first value is not confident.

    #TODO
    # rewrite for list, not pandas dataframe
    good_value = data[0]
    good_index = 0

    # loops over entirety of data
    for i in range(len(data)):
        # if confidence is over cutoff
        if confidence[i] >= p_cutoff:
            # and bad data has already been detected
            if bad_data:
                #TODO
                # Include NANs and not smooth across confidence values

                # creates a smoothed section of data
                smoothed_data = np.linspace(good_value, data[i], len(bad_index)+2)

                # injects smoothed data
                data[good_index:i+1] = smoothed_data

                # resets checking vars and flags
                bad_index = []
                bad_data = False

            # updates tracking variables
            good_value = data[i]
            good_index = i
        else:
            # flags bad data
            bad_index.append(i)
            bad_data = True

    return data, confidence

def movement_preprocess_nan(data, confidence, p_cutoff=0.60):
    """"
    Preprocesses movement data by inserting nan on low confidence data.
    """
    # loops over entirety of data
    for i in range(len(data)):
        # if confidence is not over cutoff
        if not confidence[i] >= p_cutoff:
            # insert nan
            data[i] = np.nan

    return data, confidence


def calculateDistance(x1, y1, x2, y2):
    # Function to calculate the displacement, given x and y values before and after
    # Taken from João Patriota
    if np.nan in (x1, y1, x2, y2):
        dist = np.nan
    else:
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return dist


def smoothTriangle(data, degree=3):
    # smooths data
    # Taken from João Patriota
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed = [smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def analyze_dlc_data(filepath, type="csv"):
    #TODO
    # Might need to take data directly, not the filepath
    if type == "csv":
        movement_data = pd.read_csv(filepath, low_memory=False)
    elif type == "h5":
        movement_data = pd.read_hdf(filepath, low_memory=False)
    else:
        error_message = "Don't recognise filetype"
        return error_message

    # set variables to prepare loop
    headers = list(movement_data.columns)[1:]
    tracked_parts = round(len(headers) / 3)

    confidence_array = np.array(([], []))
    x_shape = len(movement_data[2:])
    shape = (x_shape + 1, tracked_parts)
    movement_array = np.zeros(shape)

    # loop over all tracked body parts
    for i in range(tracked_parts):
        # extracts movement data
        x = movement_data[headers[i * 3]][2:].astype(float)
        y = movement_data[headers[i * 3 + 1]][2:].astype(float)
        confidence = movement_data[headers[i * 3 + 2]][2:].astype(float)

        # makes data work with numpy
        x = np.array(x)
        y = np.array(y)
        confidence = np.array(confidence)

        # smoothes data
        x = smoothTriangle(x, degree=3)
        y = smoothTriangle(y, degree=3)

        x, confidence = movement_preprocess_nan(x, confidence, p_cutoff=0.60)
        y, confidence = movement_preprocess_nan(y, confidence, p_cutoff=0.60)

        # calculates mean of confidence
        confidence_mean = confidence[2:].astype(float).mean()
        confidence_sd = confidence[2:].astype(float).std()
        confidence_array = np.append(confidence_array, np.array(([confidence_mean], [confidence_sd])), axis=1)

        # initialises variables for loop
        distance = [0]
        distance = np.array(distance)
        n_rows = len(x)

        # loops over all frames and calculates movement per frame
        for j in range(1, n_rows):
            x1 = x[j - 1]
            x2 = x[j]
            y1 = y[j - 1]
            y2 = y[j]

            # calculates distance travelled between 2 frames
            distance_frame = calculateDistance(x1, y1, x2, y2)

            # if distance travelled is low, inserts 0 as distance moved
            if distance_frame <= MIN_MOVEMENT or distance_frame == np.nan:
                distance = np.append(distance, np.nan)
            else:
                distance = np.append(distance, distance_frame)

        # inputs movement data into central array
        print(f"distance is {len(distance)}")
        print(f"movement array is {len(movement_array)}")

        # I'm getting a bug where distance is 1 value short to fit into the array, fixing it like this.
        # Very untidy and #TODO
        try:
            movement_array[:, i] = distance
        except ValueError:
            distance = np.append(distance, 0)
            movement_array[:, i] = distance

    return confidence_array, movement_array

def drive_selection(drive1x, drive1y, drive1conf, drive2x, drive2y, drive2conf, p_cutoff=0.60):
    """"
    Calculates movement of two points and selects either average of both if confidence is high,
    or takes the higher of the two. Returns an array of movement data.
    """
    drive_movement = [0] * len(drive1x)

    for j in range(1, len(drive1x)):
        if drive1conf[j] > p_cutoff and drive2conf[j] > p_cutoff:
            x1 = drive1x[j-1]
            x2 = drive1x[j]
            y1 = drive1y[j-1]
            y2 = drive1y[j]
            distance_drive1 = calculateDistance(x1, y1, x2, y2)

            x1 = drive2x[j - 1]
            x2 = drive2x[j]
            y1 = drive2y[j - 1]
            y2 = drive2y[j]
            distance_drive2 = calculateDistance(x1, y1, x2, y2)

            distance_drive = (distance_drive1 + distance_drive2) / 2

        elif drive1conf[j] > p_cutoff:
            x1 = drive1x[j - 1]
            x2 = drive1x[j]
            y1 = drive1y[j - 1]
            y2 = drive1y[j]

            distance_drive = calculateDistance(x1, y1, x2, y2)

        elif drive2conf[j] > p_cutoff:
            x1 = drive2x[j - 1]
            x2 = drive2x[j]
            y1 = drive2y[j - 1]
            y2 = drive2y[j]

            distance_drive = calculateDistance(x1, y1, x2, y2)

        else:
            distance_drive = np.nan


        drive_movement[j] = distance_drive

    drive_movement = np.array(drive_movement)
    return drive_movement
