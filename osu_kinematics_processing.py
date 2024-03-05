# OSU PROCESSING - Kayla Russell-Bertucci ()
# INPUTS = raw calibration trials and task files
# PURPOSE
# 1. rotates to ISB
# 2. Recreates markers using LCS
# 3. Filters and pads data
# OUTPUT = Processed task trials (individual trial ranges and averaged ranges)
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D


# input subject
sub_num = "S07"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"
# sample rate
fs = 100

# CALIBRATION

# set cal file folder
cal_file = r"C:\Users\kruss\OneDrive - University of Waterloo\Documents\OSU\Data\S07\Data_Raw\Trial_Kinematic\Digitized_TSV/d_S07_CAL_2.tsv"
# reads csv into a data frame
cal_raw = pd.read_csv(cal_file, sep='\t', header = 13)
# sets the row for the cal trial (note: frame#-1)
cal_frame = 5
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
        #QTM conventions: +X = forward, +Y = up, +Z = right
        #ISB conventions: +X = forward, +Y = up, +Z = right
ISB_X = np.array([0, 1, 0])
ISB_Y = np.array([0, 0, 1])
ISB_Z = np.array([-1, 0, 0])
ISB = [ISB_X, ISB_Y, ISB_Z]



def rotate_vector(vector, rotation_matrix):
    #function to rotate the marker vectors by the ISB matrix
    '''  Rotate marker vectors by the rotation matrix.
    
        Arg:
            vector: (marker) vector to be rotated
            rotation_matrix: rotation matrix (ISB in kinematics case)
    '''
    vector = np.squeeze(vector) #ensure vector is 1D
    return np.dot(rotation_matrix, vector)


''' If values need to be rotated then uncomment below '''
# # Apply rotation to cal_markers
# for marker_name, marker_values in cal_markers.items():
#     cal_markers[marker_name] = rotate_vector(marker_values, ISB)
# # Apply rotation to cal_clusters
# for cluster_name, cluster_values in cal_clusters.items():
#     cal_clusters[cluster_name] = rotate_vector(cluster_values, ISB)

#print(cal_markers)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('CAL_isb')

cal_markers_arr = []
#Create array of the dictionari
for values in cal_markers.values():
    cal_markers_arr.append(values)
cal_markers_arr = np.array(cal_markers_arr)

print(cal_markers_arr)

# fig = px.scatter_3d(x=cal_markers_arr[:,0], y = cal_markers_arr[:,1], z = cal_markers_arr[:,2])
# fig.show()


''' VISUAL CHECK OF MARKERS '''
# Function to add scatter plot and text annotations
def add_scatter_and_text(ax, marker_name, markers_dict):
    ''' Visual check of markers '''
    x, y, z = markers_dict[marker_name]
    ax.scatter(x, y, z, c='r', marker='o', label=marker_name)
    ax.text(x, y, z, f'{marker_name}', color='r')
    ax.set_xlim(0, 1500)
    ax.set_ylim(0,1500)
    ax.set_zlim(0,1500)
    origin = [0, 0, 0]
 # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set legend
ax.legend()
for marker_name in cal_markers:
    add_scatter_and_text(ax, marker_name, cal_markers)

# plt.show()
# print(cal_markers)
# Define Cal markers
# mcp2_cal = cal_raw.iloc[cal_frame,

# USE FOR TRIALS

# for cal_frame in range(len(cal_raw)):
#     # Update cal_markers for each marker
#     for marker_name, marker_columns in cal_markers.items():
#         marker_values = cal_raw.iloc[cal_frame, marker_columns].values
#         cal_markers[marker_name] = rotate_vector(marker_values, ISB)

# # Convert cal_markers dictionary to a data frame
# cal_markers_df = pd.DataFrame(cal_markers)  # Transpose for better orientation
# cal_markers_df.columns = ['X', 'Y', 'Z']  # Rename columns if needed

fa1 = cal_raw.iloc[cal_frame, 12:15].values
fa2 = cal_raw.iloc[cal_frame, 15:18].values
fa3 = cal_raw.iloc[cal_frame, 18:21].values
ua1 = cal_raw.iloc[cal_frame, 27:30].values
ua2 = cal_raw.iloc[cal_frame, 30:33].values
ua3 = cal_raw.iloc[cal_frame, 33:36].values
chest1 = cal_raw.iloc[cal_frame, 45:48].values
chest2 = cal_raw.iloc[cal_frame, 48:51].values
chest3 = cal_raw.iloc[cal_frame, 51:54].values
chest4 = cal_raw.iloc[cal_frame, 54:57].values
chest5 = cal_raw.iloc[cal_frame, 57:60].values


# Define the Local Coordinate System of the Clusters during calibration trial

    # Chest
chest_z = (chest5-chest4)/np.linalg.norm(chest5-chest4)
chest_temp = (chest1-chest5)/np.linalg.norm(chest1-chest5)
chest_x = np.cross(chest_temp, chest_z)/np.linalg.norm(np.cross(chest_temp, chest_z))
chest_y = np.cross(chest_z, chest_x)/np.linalg.norm(np.cross(chest_z, chest_x))

    # Right Upper Arm
ua_y = (ua1 - ua3) / np.linalg.norm(ua1-ua3)
ua_temp = (ua1 - ua2) / np.linalg.norm(ua1 - ua2)
ua_x = np.cross(ua_y, ua_temp) / np.linalg.norm(np.cross(ua_y, ua_temp))
ua_z = np.cross(ua_x, ua_y) / np.linalg.norm(np.cross(ua_x, ua_y))

    #Right Forearm
fa_y = (fa1 - fa3) / np.linalg.norm(fa1-fa3)
fa_temp = (fa2 - fa3) / np.linalg.norm(fa2 - fa3)
fa_z = np.cross(fa_y, fa_temp) / np.linalg.norm(np.cross(fa_y, fa_temp))
fa_x = np.cross(fa_y, fa_z) / np.linalg.norm(np.cross(fa_y, fa_z))

    # Right hand
# y = towards wrist, z = towards thumn, x = towards palm




# Vector Plotting
origin = [0, 0 ,0]
ax.quiver(*origin, *fa_x, color='r', label='Chest X')
ax.quiver(*origin, *fa_y, color='g', label='Chest Y')
ax.quiver(*origin, *fa_z, color='b', label='Chest Z')

    # Set plot limits
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

 # Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

    # Set legend
ax.legend()

    # Show the plot
plt.show()

