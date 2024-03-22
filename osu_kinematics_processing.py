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
from mpl_toolkits.mplot3d import Axes3D
import glob
import plotly.express as px



# input subject
sub_num = "S07"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"
# sample rate
fs = 100

# IMPORT CALIBRATION TRIAL AND DATA ##

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


## IMPORT TASK TRIAL AND DATA ##



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
# for marker_name in cal_markers:
#     add_scatter_and_text(ax, marker_name, cal_markers)

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

cal_fa1 = cal_raw.iloc[cal_frame, 12:15].values
cal_fa2 = cal_raw.iloc[cal_frame, 15:18].values
cal_fa3 = cal_raw.iloc[cal_frame, 18:21].values
cal_ua1 = cal_raw.iloc[cal_frame, 27:30].values
cal_ua2 = cal_raw.iloc[cal_frame, 30:33].values
cal_ua3 = cal_raw.iloc[cal_frame, 33:36].values
cal_chest1 = cal_raw.iloc[cal_frame, 45:48].values
cal_chest2 = cal_raw.iloc[cal_frame, 48:51].values
cal_chest3 = cal_raw.iloc[cal_frame, 51:54].values
cal_chest4 = cal_raw.iloc[cal_frame, 54:57].values
cal_chest5 = cal_raw.iloc[cal_frame, 57:60].values


# Define the Local Coordinate System of the Clusters during calibration trial

    # Chest
chest_z = (cal_chest4-cal_chest5)/np.linalg.norm(cal_chest4-cal_chest5)
chest_temp = (cal_chest2-cal_chest5)/np.linalg.norm(cal_chest2-cal_chest5)
chest_x = np.cross(chest_temp, chest_z)/np.linalg.norm(np.cross(chest_temp, chest_z))
chest_y = np.cross(chest_z, chest_x)/np.linalg.norm(np.cross(chest_z, chest_x))

    # Right Upper Arm
ua_y = (cal_ua3 - cal_ua1) / np.linalg.norm(cal_ua3-cal_ua1)
ua_temp = (cal_ua2 - cal_ua1) / np.linalg.norm(cal_ua2 - cal_ua1)
ua_z = np.cross(ua_y, ua_temp) / np.linalg.norm(np.cross(ua_y, ua_temp))
ua_x = np.cross(ua_y, ua_z) / np.linalg.norm(np.cross(ua_y, ua_z))

    #Right Forearm
fa_y = (cal_fa3 - cal_fa1) / np.linalg.norm(cal_fa3-cal_fa1)
fa_temp = (cal_fa2 - cal_fa1) / np.linalg.norm(cal_fa2 - cal_fa1)
fa_x = np.cross(fa_y, fa_temp) / np.linalg.norm(np.cross(fa_y, fa_temp))
fa_z = np.cross(fa_y, fa_x) / np.linalg.norm(np.cross(fa_y, fa_x))

    # Right hand
# y = towards wrist, z = towards thumb, x = towards palm
hand_y = ((np.array(cal_markers["rs"]) - np.array(cal_markers["mcp2"]))) / np.linalg.norm(np.array(cal_markers["rs"]) - np.array(cal_markers["mcp2"]))
hand_temp = ((np.array(cal_markers["us"]) - np.array(cal_markers["mcp2"]))) / np.linalg.norm(np.array(cal_markers["us"]) - np.array(cal_markers["mcp2"]))
hand_x = np.cross(hand_temp, hand_y) / np.linalg.norm(np.cross(hand_temp, hand_y))
hand_z = np.cross(hand_y, hand_x) / np.linalg.norm(np.cross(hand_y, hand_x))

# # Vector Plotting
# origin = [0, 0 ,0]
# ax.quiver(*origin, *hand_x, color='r', label='Chest X')
# ax.quiver(*origin, *hand_y, color='g', label='Chest Y')
# ax.quiver(*origin, *hand_z, color='b', label='Chest Z')
# # Set plot limits
# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
# ax.set_zlim([-10, 10])
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.legend()
# # plt.show()

# Define global coordinate axes
global_x = np.array([1, 0, 0])
global_y = np.array([0, 1, 0])
global_z = np.array([0, 0, 1])

# Compute rotation matrix to go from GCS to Cluster LCS

global_cal_chest = np.array([
    [np.dot(chest_x, global_x), np.dot(chest_x, global_y), np.dot(chest_x, global_z)],
    [np.dot(chest_y, global_x), np.dot(chest_y , global_y), np.dot(chest_y , global_z)],
    [np.dot(chest_z, global_x), np.dot(chest_z, global_y), np.dot(chest_z, global_z)],
])

global_cal_ua = np.array([
    [np.dot(ua_x, global_x), np.dot(ua_x, global_y), np.dot(ua_x, global_z)],
    [np.dot(ua_y, global_x), np.dot(ua_y , global_y), np.dot(ua_y , global_z)],
    [np.dot(ua_z, global_x), np.dot(ua_z, global_y), np.dot(ua_z, global_z)],
])

global_cal_fa = np.array([
    [np.dot(fa_x, global_x), np.dot(fa_x, global_y), np.dot(fa_x, global_z)],
    [np.dot(fa_y, global_x), np.dot(fa_y , global_y), np.dot(fa_y , global_z)],
    [np.dot(fa_z, global_x), np.dot(fa_z, global_y), np.dot(fa_z, global_z)],
])

global_cal_hand = np.array([
    [np.dot(hand_x, global_x), np.dot(hand_x, global_y), np.dot(hand_x, global_z)],
    [np.dot(hand_y, global_x), np.dot(hand_y , global_y), np.dot(hand_y , global_z)],
    [np.dot(hand_z, global_x), np.dot(hand_z, global_y), np.dot(hand_z, global_z)],
])

# Define the relationship between Clusters and markers
# Chest to SS, XP, C7, chest origin set to chest4 (bottom left)
ss_v_chest = np.dot(global_cal_chest, np.array(cal_markers["ss"]) - np.array(cal_chest4))
xp_v_chest = np.dot(global_cal_chest, np.array(cal_markers["xp"]) - np.array(cal_chest4))
c7_v_chest = np.dot(global_cal_chest, np.array(cal_markers["c7"]) - np.array(cal_chest4))

# Upper Arm to ME, LE, ua origin set to ua1
me_v_ua = np.dot(global_cal_ua, np.array(cal_markers["me"]) - np.array(cal_ua1))
le_v_ua = np.dot(global_cal_ua, np.array(cal_markers["le"]) - np.array(cal_ua1))
r_acr_v_ua = np.dot(global_cal_ua, np.array(cal_markers["r_acr"]) - np. array(cal_ua1))

# Forearm to US, RS
us_v_fa = np.dot(global_cal_fa, np.array(cal_markers["us"]) - np.array(cal_fa1))
rs_v_fa = np.dot(global_cal_fa, np.array(cal_markers["rs"]) - np.array(cal_fa1))
me_v_fa = np.dot(global_cal_ua, np.array(cal_markers["me"]) - np.array(cal_fa1))
le_v_fa = np.dot(global_cal_ua, np.array(cal_markers["me"]) - np.array(cal_fa1))

print(f'the vector from chest to ss in GCS is {ss_v_chest}')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all cal markers 
ax.scatter(cal_markers["ss"][0], cal_markers["ss"][1], cal_markers["ss"][2], c='m', marker='o', label='SS')
ax.scatter(cal_markers["xp"][0], cal_markers["xp"][1], cal_markers["xp"][2], c='m', marker='o', label='XP')
ax.scatter(cal_markers["c7"][0], cal_markers["c7"][1], cal_markers["c7"][2], c='m', marker='o', label='C7')
ax.scatter(cal_markers["r_acr"][0], cal_markers["r_acr"][1], cal_markers["r_acr"][2], c='m', marker='o', label='R_ACR')
ax.scatter(cal_markers["l_acr"][0], cal_markers["l_acr"][1], cal_markers["l_acr"][2], c='m', marker='o', label='L_ACR')
ax.scatter(cal_markers["me"][0], cal_markers["me"][1], cal_markers["me"][2], c='m', marker='o', label='me')
ax.scatter(cal_markers["le"][0], cal_markers["le"][1], cal_markers["le"][2], c='m', marker='o', label='le')
ax.scatter(cal_markers["rs"][0], cal_markers["rs"][1], cal_markers["rs"][2], c='m', marker='o', label='rs')
ax.scatter(cal_markers["us"][0], cal_markers["us"][1], cal_markers["us"][2], c='m', marker='o', label='us')
ax.scatter(cal_markers["mcp2"][0], cal_markers["mcp2"][1], cal_markers["mcp2"][2], c='m', marker='o', label='mcp2')
ax.scatter(cal_markers["mcp5"][0], cal_markers["mcp5"][1], cal_markers["mcp5"][2], c='m', marker='o', label='mcp5')



ax.legend()

# ax.scatter(XP[0, 0], XP[0, 1], XP[0, 2], c='m', marker='o', label='XP')
# ax.scatter(L5[0, 0], L5[0, 1], L5[0, 2], c='m', marker='o', label='L5')
# ax.scatter(C7[0, 0], C7[0, 1], C7[0, 2], c='m', marker='o', label='C7')
# ax.scatter(T8[0, 0], T8[0, 1], T8[0, 2], c='m', marker='o', label='T8')

# Plot additional markers
# ... (Repeat the pattern for other markers)

# Plot lines
# ax.plot([XP[0, 0], SS[0, 0]], [XP[0, 1], SS[0, 1]], [XP[0, 2], SS[0, 2]], color='black')
# ... (Repeat the pattern for other lines)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-2000, 2000])
ax.set_ylim([-2000, 2000])
ax.set_zlim([-2000, 2000])
# Show the plot
# plt.show()

''' TASK '''
# add in condition dictionaries from EMG processing 
#folder_prefix = f'd_{sub_num}_{condition}*.tsv'
# reads csv into a data frame
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV" 
trial_file = f"{trial_folder}/d_{sub_num}"
trial_raw = pd.read_csv(trial_file, sep='\t', header = 13) #sets csv to df


trial_markers = {
    "mcp2" : trial_raw.iloc[0:3].values,
    "mcp5" : trial_raw.iloc[3:6].values,
    "rs" : trial_raw.iloc[:9].values,
    "us" : trial_raw.iloc[:12].values,
    "le" : trial_raw.iloc[21:24].values,
    "me" : trial_raw.iloc[24:27].values,
    "r_acr" : trial_raw.iloc[36:39].values,
    "ss" : trial_raw.iloc[42:45].values,
    "xp" : trial_raw.iloc[60:63].values,
    "c7" : trial_raw.iloc[39:42].values,
    "l_acr" : trial_raw.iloc[63:66].values,
    }

# create a dictionary for the clusters. cluster marker : indexed frame from file
trial_clusters = {
    "fa1": trial_raw.iloc[12:15].values,
    "fa2" : trial_raw.iloc[15:16].values,
    "fa3" : trial_raw.iloc[18:21].values,
    "ua1" : trial_raw.iloc[27:30].values,
    "ua2" : trial_raw.iloc[30:33].values,
    "ua3" : trial_raw.iloc[33:36].values,
    "chest1" : trial_raw.iloc[45:48].values,
    "chest2" : trial_raw.iloc[48:51].values,
    "chest3" : trial_raw.iloc[51:54].values,
    "chest4" : trial_raw.iloc[54:57].values,
    "chest5" : trial_raw.iloc[57:60].values,
}

# Define trial cluster indexing
# (r = frames, column = 3), dataframe
trial_mcp2 = trial_raw.iloc[0:3].values
trial_mcp5 = trial_raw.iloc[3:6].values
trial_rs = trial_raw.iloc[:9].values
trial_us = trial_raw.iloc[:12].values
trial_fa1 = trial_raw.iloc[:,12:15].values
trial_fa2 = trial_raw.iloc[:,15:18].values
trial_fa3 = trial_raw.iloc[:,18:21].values
trial_ua1 = trial_raw.iloc[:,27:30].values
trial_ua2 = trial_raw.iloc[:,30:33].values
trial_ua3 = trial_raw.iloc[:,33:36].values
trial_chest1 = trial_raw.iloc[:,45:48].values
trial_chest2 = trial_raw.iloc[:,48:51].values
trial_chest3 = trial_raw.iloc[:,51:54].values
trial_chest4 = trial_raw.iloc[:,54:57].values
trial_chest5 = trial_raw.iloc[:,57:60].values


''' DEFINE LCS AND UNIT VECTORS FOR TASK CLUSTERS '''
# use imported task trial to iterate through cluster markers
# remaking coordinate systems using task clusters
trial_frame_count = len(trial_raw)

# create empty lists to store LCS vectors
# hand vectors
hand_trial_x = []
hand_trial_y = []
hand_trial_z = []

# forearm vectors 
fa_trial_x = []
fa_trial_y = []
fa_trial_z = []

# upper arm vectors
ua_trial_x = []
ua_trial_y = []
ua_trial_z = []

# chest vectors
chest_trial_x = []
chest_trial_y = []
chest_trial_z = []

#iterate through the length of the trial to create a LCS for each frame
for frame in range(trial_frame_count):
    #Hand, O = mcp2, Y = towards wrist, Z = radial X = dorsal
    hand_trial_y_frame = ((trial_rs[frame, :] - trial_mcp2[frame, :])) /(np.linalg.norm(trial_rs[frame, :] - trial_mcp2[frame, :]))
    hand_trial_temp_frame = (trial_us[frame,: ] - trial_mcp2[frame, :]) / (np.linalg.norm(trial_us[frame, :] - trial_mcp2[frame, :]))
    hand_trial_z_frame = np.cross(hand_trial_y_frame, hand_trial_temp_frame) / np.linalg.norm(np.cross(hand_trial_y_frame, hand_trial_temp_frame))
    hand_trial_x_frame = np.cross(hand_trial_z_frame, hand_trial_y_frame) / np.linalg.norm(np.cross(hand_trial_z_frame, hand_trial_z_frame))
    
    #Forearm
    fa_trial_y_frame = ((trial_fa3[frame, :] - trial_fa1[frame, :])) /(np.linalg.norm(trial_fa3[frame, :] - trial_fa1[frame, :]))
    fa_trial_temp_frame = (trial_fa2[frame,: ] - trial_fa1[frame, :]) / (np.linalg.norm(trial_fa2[frame, :] - trial_fa1[frame, :]))
    fa_trial_x_frame = np.cross(fa_trial_y_frame, fa_trial_temp_frame) / np.linalg.norm(np.cross(fa_trial_y_frame, fa_trial_temp_frame))
    fa_trial_z_frame = np.cross(fa_trial_x_frame, fa_trial_y_frame) / np.linalg.norm(np.cross(fa_trial_x_frame, fa_trial_y_frame))
    #Upper arm
    ua_trial_y_frame = ((trial_ua3[frame, :] - trial_ua1[frame, :])) /(np.linalg.norm(trial_ua3[frame, :] - trial_ua1[frame, :]))
    ua_trial_temp_frame = (trial_ua2[frame, :] - trial_ua1[frame, :]) / (np.linalg.norm(trial_ua2[frame, :] - trial_ua1[frame, :]))
    ua_trial_x_frame = np.cross(ua_trial_y_frame, ua_trial_temp_frame) / np.linalg.norm(np.cross(ua_trial_y_frame, ua_trial_temp_frame))
    ua_trial_z_frame = np.cross(ua_trial_y_frame, ua_trial_x_frame) / np.linalg.norm(np.cross(ua_trial_y_frame, ua_trial_x_frame))
    #Chest
    chest_trial_z_frame = ((trial_chest4[frame, :] - trial_chest5[frame, :])) /(np.linalg.norm(trial_chest4[frame, :] - trial_chest5[frame, :]))
    chest_trial_temp_frame = (trial_chest2[frame, :] - trial_chest5[frame, :]) / (np.linalg.norm(trial_chest2[frame, :] - trial_chest5[frame, :]))
    chest_trial_x_frame = np.cross(chest_trial_z_frame, chest_trial_temp_frame) / np.linalg.norm(np.cross(chest_trial_z_frame, chest_trial_temp_frame))
    chest_trial_y_frame = np.cross(chest_trial_z_frame, chest_trial_x_frame) / np.linalg.norm(np.cross(chest_trial_z_frame, chest_trial_x_frame))

# ******* 5. Define rotation matrix (global to local) for task clusters *******
    
# # set empty arrays for rotation matrix of each LCS
# hand_trial_GRL = np.zeros((3,3))
# fa_trial_GRL = np.zeros((3,3))    
# ua_trial_GRL = np.zeros((3,3))
# chest_trial_GRL = np.zeros((3,3))

def compute_GRL_rotation_matrix(LCS_v_x, LCS_v_y, LCS_v_z):
    """ Computes a Global to Local rotation matrix
    Inputs
    LCS_v_x = x vector of the LCS
    LCS_v_y = y vector of the LCS
    LCS_v_z = z vector of the LCS
    """
    trial_global_rot = np.zeros((3,3))

    trial_global_rot[0,0] = np.dot(LCS_v_x, global_x)
    trial_global_rot[0,1] = np.dot(LCS_v_x, global_y)
    trial_global_rot[0,2] = np.dot(LCS_v_x, global_z)
    trial_global_rot[1,0] = np.dot(LCS_v_y, global_x)
    trial_global_rot[1,1] = np.dot(LCS_v_y, global_y)
    trial_global_rot[1,2] = np.dot(LCS_v_y, global_z)
    trial_global_rot[2,0] = np.dot(LCS_v_z, global_x)
    trial_global_rot[2,1] = np.dot(LCS_v_z, global_y)
    trial_global_rot[2,2] = np.dot(LCS_v_z, global_z)

    return trial_global_rot


# iterate through 
for frame in range(trial_frame_count):
    hand_trial_GRL = compute_GRL_rotation_matrix(hand_trial_x_frame, hand_trial_y_frame, hand_trial_z_frame)
    fa_trial_GRL = compute_GRL_rotation_matrix(fa_trial_x_frame, fa_trial_y_frame, fa_trial_z_frame)
    ua_trial_GRL = compute_GRL_rotation_matrix(ua_trial_x_frame, ua_trial_y_frame, ua_trial_z_frame)
    chest_trial_GRL = compute_GRL_rotation_matrix(chest_trial_x_frame, chest_trial_y_frame, chest_trial_z_frame)

#tried to make for loop but idk how to incorporate it oops
# for condition, values in condition_names.items():
#     #sets folder prefix to obtain files
#     folder_prefix = f'd_{sub_num}_{condition}*.tsv'
#     # groups trial paths based on their folder prefix containing condition name
#     condition_trial_paths = glob.glob(f'{task_folder}/{folder_prefix}')
#     if len(condition_trial_paths) == 0: continue #doesnt break for low lvl
#     for condition_trial in condition_trial_paths:
    

    # dictionary with each condition name and empty values 
condition_names = {
    "EASY_PREF": [],
    "EASY_HIGH": [],
    "EASY_LOW": [],
    "HARD_PREF": [],
    "HARD_HIGH": [],
    "HARD_LOW": [],
}

