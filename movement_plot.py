import numpy as np
import pandas as pd
import math
import pickle
import os

# set constants
EPOCH_LENGTH = 120
P_CUTOFF = 0.60
MIN_MOVEMENT = 2


def movement_preprocess(data, confidence, p_cutoff=0.60):
    """"
    Preprocesses movement data by smoothing across intervals of low confidence.
    """
    #TODO
    # Give warnings if high amount of bad intervals (2s or up)
    bad_data = False
    bad_index = []

    #TODO fix this bug
    # this is not good, will cause very bad data in samples where first value is not confident.

    good_value = data.iloc[0]
    good_index = 0

    # testing var
    blah = 0

    # loops over entirety of data
    for i in range(len(data)):
        # if confidence is over cutoff
        if confidence.iloc[i] >= p_cutoff:
            # and bad data has already been detected
            if bad_data:
                #TODO
                # Include NANs and not smooth across confidence values

                # creates a smoothed section of data
                smoothed_data = np.linspace(good_value, data.iloc[i], len(bad_index)+2)

                # injects smoothed data
                data.iloc[good_index:i+1] = smoothed_data

                # resets checking vars and flags
                bad_index = []
                bad_data = False

            # updates tracking variables
            good_value = data.iloc[i]
            good_index = i
        else:
            # flags bad data
            bad_index.append(i)
            bad_data = True

    return data, confidence


def calculateDistance(x1,y1,x2,y2):
    # Function to calculate the displacement, given x and y values before and after
    # Taken from João Patriota
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
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


recording = "fear_awaker120DLC_resnet50_1stBatch_behaviourJun18shuffle1_900000.csv"
# load in DLC data
movement_data = pd.read_csv(recording, low_memory=False)

# set variables to prepare loop
headers = list(movement_data.columns)[1:]
tracked_parts = round(len(headers) / 3)
n_epochs = round(len(movement_data) / EPOCH_LENGTH)

confidence_array = np.array(([], []))
shape = (len(movement_data[2:]), tracked_parts)
movement_array = np.zeros(shape)

# loop over all tracked body parts
for i in range(tracked_parts):
    # extracts movement data
    x = movement_data[headers[i * 3]][2:].astype(float)
    y = movement_data[headers[i * 3 + 1]][2:].astype(float)
    confidence = movement_data[headers[i * 3 + 2]]

    # makes data work with numpy
    x = np.array(x)
    y = np.array(y)

    # smoothes data
    # TODO
    # Do we want to both interpolate the data and then smooth it? Seems like a double pass
    x = smoothTriangle(x, degree=3)
    y = smoothTriangle(y, degree=3)

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
        if distance_frame <= MIN_MOVEMENT:
            distance = np.append(distance, 0)
        else:
            distance = np.append(distance, distance_frame)

    # inputs movement data into central array
    movement_array[:, i] = distance

plotting_array = []
for j in range(n_epochs):
    pass
