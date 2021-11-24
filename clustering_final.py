import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from movement_util import *
from plotting_util import *

N_CLUSTERS = 4
N_METRICS = 9

# loads in LFP metrics per area split into epochs
filename = "pickleFiles/LFP_power_areas.pickle"
LFP_metrics = open(filename, 'rb')
LFP_power_areas = pickle.load(LFP_metrics)
LFP_metrics.close()

# loads in epoch division
filename = "pickleFiles/epoch_frames_all.pickle"
epoch_frames_file = open(filename, 'rb')
epoch_frames_all = pickle.load(epoch_frames_file)
epoch_frames_file.close()

movement_filepath = "fear_sleepr120DLC_resnet50_DriveNetwork2Dec15shuffle1_1030000.csv"
movement_conf, movement_data = analyze_dlc_data(movement_filepath, type="csv")

# extracts movement data, triangle of drive left/right and back
back_data = movement_data[:, -1]

# prepares for drive selection, #TODO this is hard coded
movement_data_raw = pd.read_csv(movement_filepath, low_memory=False)
headers = list(movement_data_raw.columns)[1:]

drive1x = np.array(movement_data_raw[headers[12]][2:].astype(float))
drive1y = np.array(movement_data_raw[headers[13]][2:].astype(float))
drive1conf = np.array(movement_data_raw[headers[14]][2:].astype(float))
drive2x = np.array(movement_data_raw[headers[15]][2:].astype(float))
drive2y = np.array(movement_data_raw[headers[16]][2:].astype(float))
drive2conf = np.array(movement_data_raw[headers[17]][2:].astype(float))

# Determines movement of drives
drive_movement = drive_selection(drive1x, drive1y, drive1conf, drive2x, drive2y, drive2conf)

# levels back and drive movement
while len(back_data) != len(drive_movement):
    if len(back_data) > len(drive_movement):
        drive_movement = np.append(drive_movement, 0)
    else:
        back_data = np.append(back_data, 0)

# takes average measure of drive and back movement
shape = (len(back_data), 2)
back_drive_movement = np.zeros(shape)

back_drive_movement[:, 0] = back_data
back_drive_movement[:, 1] = drive_movement

# summarises back and drive movement as average
back_drive_movement = (np.nansum(back_drive_movement, axis=1) / 2)


# pre allocates var for data storage
movement_data_processed = np.zeros((len(epoch_frames_all), 2))
back_drive_movement_epochd = np.zeros((len(epoch_frames_all), 1))

# determines movement measures per epoch
for i in range(0, len(epoch_frames_all)):
    # determines start and end frame of epoch
    start_frame = epoch_frames_all[i][0]
    end_frame = epoch_frames_all[i][1]
    len_epoch = end_frame - start_frame

    # determines average movement for right drive within frames of epoch
    movement_data_processed[i, 0] = np.nansum(back_data[start_frame: end_frame]) / len_epoch
    movement_data_processed[i, 1] = np.nansum(drive_movement[start_frame: end_frame]) / len_epoch
    back_drive_movement_epochd[i] = np.nansum(back_drive_movement[start_frame: end_frame]) / len_epoch

# creates new fit object and clusters
#TODO
# issues with epoch saving: missing 20, exact amount added to epoch_frames, might  be an issue with last 20.
fit_object = np.append(LFP_power_areas, back_drive_movement_epochd[0:5800, :], axis=1)

# performs a fit of the data
gmm = GaussianMixture(n_components=N_CLUSTERS, init_params='kmeans', verbose=0).fit(fit_object)

# assigns labels to each epoch
labels = gmm.fit_predict(fit_object)

# General Gaussian Mixture clustering including all data: edit code in terminal and see what sticks, keep notes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

fit_object_scaled = scaler.fit_transform(fit_object)
gmm = GaussianMixture(n_components=N_CLUSTERS, init_params='kmeans', verbose=0).fit(fit_object_scaled)

labels_scaled = gmm.fit_predict(fit_object_scaled)

hc_theta = fit_object_scaled[:, 0]
hc_delta = fit_object_scaled[:, 1]

ac_theta = fit_object_scaled[:, 2]
ac_delta = fit_object_scaled[:, 3]

bla_theta = fit_object_scaled[:, 4]
bla_delta = fit_object_scaled[:, 5]

mpfc_theta = fit_object_scaled[:, 6]
mpfc_delta = fit_object_scaled[:, 7]

movement_scaled = fit_object_scaled[:, 8]

# different plot combinations
plot_scatter(hc_delta, hc_theta, labels=labels_scaled, title="HC delta vs HC theta, scaled",
                           xlabel="HC delta power(log)", ylabel="HC theta power(log)")

plot_scatter(mpfc_theta, bla_theta, labels=labels_scaled, title="MPFC theta vs BLA theta, scaled",
                           xlabel="MPFC theta power(log)", ylabel="BLA theta power(log)")

plot_scatter(bla_delta, bla_theta, labels=labels_scaled, title="BLA delta vs. BLA theta",
                           xlabel="BLA delta", ylabel="BLA theta")

plt.close()
##############################
#
# State classification
#
##############################


data_per_cluster = [[]] * len(set(labels_scaled))
# pulls some metrics from the data for general purpose
for i in range(len(set(labels_scaled))):
    # selects epochs within selected cluster
    mask = np.where(labels_scaled == i)[0]

    # gets metrics
    cluster_data = fit_object[mask, :]

    # stores metrics for later use
    data_per_cluster[i] = cluster_data

# preparing vars for plotting
plot_pos = list(range(N_CLUSTERS))
plot_data = [[]] * N_CLUSTERS

plot_titles = ["Hippocampal Delta (log)",
              "Hippocampal Theta (log)",
              "Auditory Cortex Delta (log)",
              "Auditory Cortex Theta (log)",
              "Basal Lateral Amygdala Delta (log)",
              "Basal Lateral Amygdala Theta (log)",
              "medial Prefrontal Cortex Delta (log)",
              "medial Prefrontal Cortex Theta (log)",
              "Movement, back and drive (au)"]

# creates var for mean/std saving
values = np.zeros((N_METRICS, N_CLUSTERS, 2))

# for all metrics
for i in range(N_METRICS):
    # and all clusters
    for j in range(N_CLUSTERS):
        # pull data from saving var
        data = data_per_cluster[j][:, i]
        plot_data[j] = data

        # gets means and deviations from data for later reference/classification
        values[i, j, 0] = np.nanmean(data)
        values[i, j, 1] = np.nanstd(data)

    # create a boxplot of all 4 clusters for the same metric
    plt.boxplot(plot_data, positions=plot_pos)
    plt.title(plot_titles[i] + " per cluster")
    plt.ylabel(plot_titles[i])
    plt.savefig(plot_titles[i])
    plt.close()


# Create a hypnogram

# creates axis to be plotted against
hypnogram_axis = np.linspace(0, len(labels_scaled) - 1, num=len(labels_scaled), dtype=int)

plt.plot()



""""
    for j in range(N_CLUSTERS):
        data = data_per_cluster[j]
        plt.boxplot(data[:, i], positions=plot_pos[j])    
    """

""""
# DBSCAN testing
fit_object_no_back = np.delete(fit_object, 8, 1)
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=3, min_samples=1).fit(fit_object_no_back)

labels_db = db.labels_
n_labels = len(set(labels_db)) - (1 if -1 in labels_db else 0)

# GGM no movement at all
# performs a fit of the data
fit_object_no_movement = np.delete(fit_object, (8, 9), 1)
gmm_no_movement = GaussianMixture(n_components=4, init_params='kmeans', verbose=0).fit(fit_object_no_movement)
labels_no_movement = gmm_no_movement.fit_predict(fit_object_no_movement)
"""

from sklearn.cluster import OPTICS
# sets vars for OPTICS algorithm
min_samples = 5
xi=0.05
min_cluster_size = 0.02

# creates clustering algorithm and fits data to it
clustering = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size).fit(fit_object)

# gets amount of labels
labels_OPTICS = clustering.labels_

for label in set(labels_OPTICS):
    print(f"There are {len(np.where(labels_OPTICS == label)[0])} samples in cluster {label}")

cluster_mask = np.where(labels_OPTICS == 0)[0]

# very small cluster for one, remove those samples as undefined and recluster
clustering = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size).fit(fit_object[cluster_mask])

# gets amount of labels
labels_OPTICS = clustering.labels_

for label in set(labels_OPTICS):
    print(f"There are {len(np.where(labels_OPTICS == label)[0])} samples in cluster {label}")
