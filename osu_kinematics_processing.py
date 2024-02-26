# OSU PROCESSING - Kayla Russell-Bertucci ()
# INPUTS = raw calibration trials and task files
# PURPOSE
# 1. rotates to ISB
# 2. Recreates markers using LCS
# 3. Filters and pads data
# OUTPUT = Processed task trials (individual trial ranges and averaged ranges)

import pandas as pd
import numpy as np
import glob

# input subject
sub_num = "S05"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"

# sample rate
fs = 100

# CALIBRATION

# set cal file folder
cal_file = f"{trial_folder}/d_{sub_num}_CAL_2.tsv"
# reads csv into a data frame
cal_raw = pd.read_csv(cal_file, sep='\t', header = 12)
# sets the row for the cal trial (note: frame#-1)
cal_frame = 3
# create dictionary containing marker names and their columns

#create a dictionary for the markers, marker name : indexed lines from the file
cal_markers = {
    "mcp2" : cal_raw.iloc[cal_frame, 0:3].values,
    "mcp5" : cal_raw.iloc[cal_frame,3:6].values,
    "rs" : cal_raw.iloc[cal_frame,6:9].values,
    "us" : cal_raw.iloc[cal_frame,9:12].values,
    "le" : cal_raw.iloc[cal_frame, 21:24].values,
    "me" : cal_raw.iloc[cal_frame, 24:27].values,
    "r_acr" : cal_raw.iloc[cal_frame, 36:39].values,
    "ss" : cal_raw.iloc[cal_frame, 42:45].values,
    "xp" : cal_raw.iloc[cal_frame, 60:63].values,
    "c7" : cal_raw.iloc[cal_frame, 39:42].values,
    "l_acr" : cal_raw.iloc[cal_frame, 63:66].values,
    }
# create a dictionary for the clusters. cluster marker : indexed frame from file
cal_clusters = {
    "fa1": cal_raw.iloc[cal_frame, 12:15].values,
    "fa2" : cal_raw.iloc[cal_frame, 15:16].values,
    "fa3" : cal_raw.iloc[cal_frame, 18:21].values,
    "ua1" : cal_raw.iloc[cal_frame, 27:30].values,
    "ua2" : cal_raw.iloc[cal_frame, 30:33].values,
    "ua3" : cal_raw.iloc[cal_frame, 33:36].values,
    "chest1" : cal_raw.iloc[cal_frame, 45:48].values,
    "chest2" : cal_raw.iloc[cal_frame, 48:51].values,
    "chest3" : cal_raw.iloc[cal_frame, 51:54].values,
    "chest4" : cal_raw.iloc[cal_frame, 54:57].values,
    "chest5" : cal_raw.iloc[cal_frame, 57:60].values,
}

# ROTATE TO ISB AXES
    #Define axes rotation matrix
        #QTM conventions: +X = , +Y = , +Z = up
        #ISB conventions: +X = forward, +Y = up, +Z = right
ISB_X = np.array([0, 1, 0])
ISB_Y = np.array([0, 0, 1])
ISB_Z = np.array([-1, 0, 0])
ISB = [ISB_X, ISB_Y, ISB_Z]

print(cal_markers)

def rotate_vector(vector, rotation_matrix):
    #function to rotate the marker vectors by the ISB matrix
    '''  Rotate marker vectors by the rotation matrix.
    
        Arg:
            vector: (marker) vector to be rotated
            rotation_matrix: rotation matrix (ISB in kinematics case)
    '''
    vector = np.squeeze(vector) #ensure vector is 1D
    return np.dot(rotation_matrix, vector)

# Apply rotation to cal_markers
for marker_name, marker_values in cal_markers.items():
    cal_markers[marker_name] = rotate_vector(marker_values, ISB)

# Apply rotation to cal_clusters
for cluster_name, cluster_values in cal_clusters.items():
    cal_clusters[cluster_name] = rotate_vector(cluster_values, ISB)

print(cal_markers)
# Define Cal markers
# mcp2_cal = cal_raw.iloc[cal_frame,

