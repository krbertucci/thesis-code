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
import scipy


# input subject
sub_num = "S01"
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


# Participants will either be standing in the same orientation during 
# cal trials or rotated about Y axis 
# if participant is standing facing the desk, do not need to rotate the virtual markers
# if participant is standing facing the door, rotate about Y
# ROTATE TO ISB AXES
    #Define axes rotation matrix
        #QTM conventions: +X = forward, +Y = up, +Z = right
        #ISB conventions: +X = forward, +Y = up, +Z = right
ISB_X = np.array([1, 0, 0])
ISB_Y = np.array([0, 1, 0])
ISB_Z = np.array([0, 0, 1])
ISB = np.array([ISB_X, ISB_Y, ISB_Z])

# If CAL facing door
    # LCS
        # +X Local = -Z Global
        # +Y Local = +Y Global
        # +Z Local = +X Global
alt_CAL_ISB_X = np.array([0, 0, 1])
alt_CAL_ISB_Y = np.array([0, 1, 0])
alt_CAL_ISB_Z = np.array([-1, 0, 0])

alt_CAL_GRL = np.array([alt_CAL_ISB_X, alt_CAL_ISB_Y, alt_CAL_ISB_Z])


def rotate_vector(vector, rotation_matrix):
    #function to rotate the marker vectors by the ISB matrix using dot product
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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('CAL_isb')

cal_markers_arr = []
#Create array of the dictionari
for values in cal_markers.values():
    cal_markers_arr.append(values)
cal_markers_arr = np.array(cal_markers_arr)



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



# USE FOR TRIALS

# for cal_frame in range(len(cal_raw)):
#     # Update cal_markers for each marker
#     for marker_name, marker_columns in cal_markers.items():
#         marker_values = cal_raw.iloc[cal_frame, marker_columns].values
#         cal_markers[marker_name] = rotate_vector(marker_values, ISB)

# cal cluster marker arrays
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

cal_ss = cal_raw.iloc[cal_frame, 42:45].values
cal_xp = cal_raw.iloc[cal_frame, 60:63].values,
cal_c7 = cal_raw.iloc[cal_frame, 39:42].values,

print(cal_ss)
cal_ss = (np.dot(ISB,cal_ss))
print(cal_ss)
# Define the Local Coordinate System of the Clusters during calibration trial

    # Chest, O = chest5, +Y = superior, +X = anterior, +Z = right 
cal_chest_z = (cal_chest4-cal_chest5)/np.linalg.norm(cal_chest4-cal_chest5)
cal_chest_temp = (cal_chest3-cal_chest5)/np.linalg.norm(cal_chest3-cal_chest5)
cal_chest_x = np.cross(cal_chest_temp, cal_chest_z)/np.linalg.norm(np.cross(cal_chest_temp, cal_chest_z))
cal_chest_y = np.cross(cal_chest_z, cal_chest_x)/np.linalg.norm(np.cross(cal_chest_z, cal_chest_x))

    # Right Upper Arm, tip facing POSTERIOR, O = ua1, +Y = superior, +X = anterior, +Z = lateral
cal_ua_y = (cal_ua3 - cal_ua1) / np.linalg.norm(cal_ua3-cal_ua1)
cal_ua_temp = (cal_ua2 - cal_ua1) / np.linalg.norm(cal_ua2 - cal_ua1)
cal_ua_z = np.cross(cal_ua_y, cal_ua_temp) / np.linalg.norm(np.cross(cal_ua_y, cal_ua_temp))
cal_ua_x = np.cross(cal_ua_y, cal_ua_z) / np.linalg.norm(np.cross(cal_ua_y, cal_ua_z))

    # Right Upper Arm, tip facing ANTERIOR, O = ua1, +Y = superior, +X = anterior, +Z = lateral
# cal_ua_y = (cal_ua3 - cal_ua1) / np.linalg.norm(cal_ua3-cal_ua1)
# cal_ua_temp = (cal_ua2 - cal_ua1) / np.linalg.norm(cal_ua2 - cal_ua1)
# cal_ua_z = np.cross(cal_ua_temp, cal_ua_y) / np.linalg.norm(np.cross(cal_ua_temp, cal_ua_y))
# cal_ua_x = np.cross(cal_ua_y, cal_ua_z) / np.linalg.norm(np.cross(cal_ua_y, cal_ua_z))

    #Right Forearm, O = fa1, y = towards elbow, z = posterior, x = lateral
cal_fa_y = (cal_fa3 - cal_fa1) / np.linalg.norm(cal_fa3-cal_fa1)
cal_fa_temp = (cal_fa2 - cal_fa1) / np.linalg.norm(cal_fa2 - cal_fa1)
cal_fa_x = np.cross(cal_fa_y, cal_fa_temp) / np.linalg.norm(np.cross(cal_fa_y, cal_fa_temp))
cal_fa_z = np.cross(cal_fa_x, cal_fa_y) / np.linalg.norm(np.cross(cal_fa_x, cal_fa_y))

    # Right hand, O = mcp2, y = towards wrist, z = towards thumb, x = towards palm
cal_hand_y = ((np.array(cal_markers["rs"]) - np.array(cal_markers["mcp2"]))) / np.linalg.norm(np.array(cal_markers["rs"]) - np.array(cal_markers["mcp2"]))
cal_hand_temp = ((np.array(cal_markers["us"]) - np.array(cal_markers["mcp2"]))) / np.linalg.norm(np.array(cal_markers["us"]) - np.array(cal_markers["mcp2"]))
cal_hand_x = np.cross(cal_hand_y, cal_hand_temp) / np.linalg.norm(np.cross(cal_hand_y, cal_hand_temp))
cal_hand_z = np.cross(cal_hand_x, cal_hand_y) / np.linalg.norm(np.cross(cal_hand_x, cal_hand_y))



# Define global coordinate axes
global_x = np.array([1, 0, 0])
global_y = np.array([0, 1, 0])
global_z = np.array([0, 0, 1])

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


# Compute rotation matrix to go from GCS to Cluster LCS
global_cal_chest = compute_GRL_rotation_matrix(cal_chest_x, cal_chest_y, cal_chest_z)
global_cal_ua = compute_GRL_rotation_matrix(cal_ua_x, cal_ua_y, cal_ua_z)
global_cal_fa = compute_GRL_rotation_matrix(cal_fa_x, cal_fa_y, cal_fa_z)
global_cal_hand = compute_GRL_rotation_matrix(cal_hand_x, cal_hand_y, cal_hand_z)


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
le_v_fa = np.dot(global_cal_ua, np.array(cal_markers["le"]) - np.array(cal_fa1))

'''TEST REMOVING RM '''
# ss_v_chest = (np.array(cal_markers["ss"]) - np.array(cal_chest4))
# xp_v_chest = (np.array(cal_markers["xp"]) - np.array(cal_chest4))
# c7_v_chest = (np.array(cal_markers["c7"]) - np.array(cal_chest4))

# # Upper Arm to ME, LE, ua origin set to ua1
# me_v_ua = (np.array(cal_markers["me"]) - np.array(cal_ua1))
# le_v_ua = (np.array(cal_markers["le"]) - np.array(cal_ua1))
# r_acr_v_ua = (np.array(cal_markers["r_acr"]) - np. array(cal_ua1))

# # Forearm to US, RS
# us_v_fa = (np.array(cal_markers["us"]) - np.array(cal_fa1))
# rs_v_fa = (np.array(cal_markers["rs"]) - np.array(cal_fa1))
# me_v_fa = (np.array(cal_markers["me"]) - np.array(cal_fa1))
# le_v_fa = (np.array(cal_markers["le"]) - np.array(cal_fa1))

def vector_plot(ax, origin, vector):
    ''' 
    plots vector
    '''
    ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='purple')


''' Visual Plotting vectors to virtual markers '''

# origin = cal_chest4
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # scatter plot for original markers
# # origins set to black
# ax.scatter((cal_markers["ss"])[0], (cal_markers["ss"])[1], (cal_markers["ss"])[2], color='steelblue', marker = 'o', label = 'ss')
# ax.scatter((cal_markers["c7"])[0], (cal_markers["c7"])[1], (cal_markers["c7"])[2], color='brown', marker = 'o', label = 'c7')
# ax.scatter((cal_markers["xp"])[0], (cal_markers["xp"])[1], (cal_markers["xp"])[2], color='green', marker = 'o', label = 'xp')
# ax.scatter(cal_chest4[0], cal_chest4[1], cal_chest4[2], color='black', marker = 'o', label = 'chest 4')

# ax.scatter(cal_ua1[0], cal_ua1[1], cal_ua1[2], color='black', marker = 'o', label = 'ua1')
# ax.scatter(cal_fa1[0], cal_fa1[1], cal_fa1[2], color='black', marker = 'o', label = 'ua1')
# ax.scatter((cal_markers["r_acr"])[0], (cal_markers["r_acr"])[1], (cal_markers["r_acr"])[2],  color='c', marker = 'o', label = 'r_acr')
# ax.scatter((cal_markers["le"])[0], (cal_markers["le"])[1], (cal_markers["le"])[2],  color='orange', marker = 'o', label = 'le')
# ax.scatter((cal_markers["me"])[0], (cal_markers["me"])[1], (cal_markers["me"])[2],  color='purple', marker = 'o', label = 'me')
# ax.scatter((cal_markers["rs"])[0], (cal_markers["rs"])[1], (cal_markers["rs"])[2],  color='magenta', marker = 'o', label = 'me')
# ax.scatter((cal_markers["us"])[0], (cal_markers["us"])[1], (cal_markers["us"])[2],  color='r', marker = 'o', label = 'me')

# #quiver plot to plot vectors between origin and original markers
# ax.quiver(cal_chest4[0], cal_chest4[1], cal_chest4[2], ss_v_chest[0], ss_v_chest[1], ss_v_chest[2], color='steelblue')
# ax.quiver(cal_chest4[0], cal_chest4[1], cal_chest4[2], xp_v_chest[0], xp_v_chest[1], xp_v_chest[2], color='green')
# ax.quiver(cal_chest4[0], cal_chest4[1], cal_chest4[2], c7_v_chest[0], c7_v_chest[1], c7_v_chest[2], color='brown')
# #acromion from ua1
# ax.quiver(cal_ua1[0], cal_ua1[1], cal_ua1[2], r_acr_v_ua[0], r_acr_v_ua[1], r_acr_v_ua[2], color='cyan')
# #me and le from ua1 (if fa cluster is proximal)
# ax.quiver(cal_ua1[0], cal_ua1[1], cal_ua1[2], me_v_ua[0], me_v_ua[1], me_v_ua[2], color='purple')
# ax.quiver(cal_ua1[0], cal_ua1[1], cal_ua1[2], le_v_ua[0], le_v_ua[1], le_v_ua[2], color='orange')
# # me, le, rs, us from fa1
# ax.quiver(cal_fa1[0], cal_fa1[1], cal_fa1[2], me_v_fa[0], me_v_fa[1], me_v_fa[2], color='purple')
# ax.quiver(cal_fa1[0], cal_fa1[1], cal_fa1[2], le_v_fa[0], le_v_fa[1], le_v_fa[2], color='orange')
# ax.quiver(cal_fa1[0], cal_fa1[1], cal_fa1[2], rs_v_fa[0], rs_v_fa[1], rs_v_fa[2], color='red')
# ax.quiver(cal_fa1[0], cal_fa1[1], cal_fa1[2], us_v_fa[0], us_v_fa[1], us_v_fa[2], color='magenta')

# ax.quiver(*cal_ua1, *cal_ua_y, color='r', label='UA Y')
# ax.quiver(*cal_ua1, *cal_ua_x, color='g', label='UA X')
# ax.quiver(*cal_ua1, *cal_ua_z, color='b', label='UA Z')


# # Set plot limits
# ax.set_xlim([-100, 1000])
# ax.set_ylim([500, 1200])
# ax.set_zlim([-0, 1000])
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.legend()
# plt.show()

''' TASK '''
# add in condition dictionaries from EMG processing 
#folder_prefix = f'd_{sub_num}_{condition}*.tsv'
# reads csv into a data frame
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV" 
trial_file = f"{trial_folder}/d_{sub_num}_EASY_LOW_4.tsv"
trial_raw = pd.read_csv(trial_file, sep='\t', header = 13) #sets csv to df

# Define trial cluster indexing
# (r = frames, column = 3), dataframe
trial_mcp2 = trial_raw.iloc[:,0:3].values
trial_mcp5 = trial_raw.iloc[:,3:6].values
trial_rs = trial_raw.iloc[:,6:9].values
trial_us = trial_raw.iloc[:,9:12].values
trial_le = trial_raw.iloc[:,21:24].values
trial_me = trial_raw.iloc[:,24:27].values
trial_fa1 = trial_raw.iloc[:,12:15].values
trial_fa2 = trial_raw.iloc[:,15:18].values
trial_fa3 = trial_raw.iloc[:,18:21].values
trial_ua1 = trial_raw.iloc[:,27:30].values
trial_ua2 = trial_raw.iloc[:,30:33].values
trial_ua3 = trial_raw.iloc[:,33:36].values
trial_racr = trial_raw.iloc[:, 36:39].values
trial_lacr = trial_raw.iloc[:, 63:66].values
trial_chest1 = trial_raw.iloc[:,45:48].values
trial_chest2 = trial_raw.iloc[:,48:51].values
trial_chest3 = trial_raw.iloc[:,51:54].values
trial_chest4 = trial_raw.iloc[:,54:57].values
trial_chest5 = trial_raw.iloc[:,57:60].values
trial_ss = trial_raw.iloc[:,42:45].values
trial_xp = trial_raw.iloc[:, 60:63].values
trial_c7 = trial_raw.iloc[:, 39:42].values



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
     #Chest
    chest_trial_z_frame = ((trial_chest4[frame, :] - trial_chest5[frame, :])) /(np.linalg.norm(trial_chest4[frame, :] - trial_chest5[frame, :]))
    chest_trial_temp_frame = (trial_chest3[frame, :] - trial_chest5[frame, :]) / (np.linalg.norm(trial_chest3[frame, :] - trial_chest5[frame, :]))
    chest_trial_x_frame = np.cross(chest_trial_temp_frame, chest_trial_z_frame) / np.linalg.norm(np.cross(chest_trial_temp_frame, chest_trial_z_frame))
    chest_trial_y_frame = np.cross(chest_trial_z_frame, chest_trial_x_frame) / np.linalg.norm(np.cross(chest_trial_z_frame, chest_trial_x_frame))
    # Right Upper Arm, tip facing POSTERIOR, O = ua1, +Y = superior, +X = anterior, +Z = lateral
    ua_trial_y_frame = ((trial_ua3[frame, :] - trial_ua1[frame, :])) /(np.linalg.norm(trial_ua3[frame, :] - trial_ua1[frame, :]))
    ua_trial_temp_frame = (trial_ua2[frame, :] - trial_ua1[frame, :]) / (np.linalg.norm(trial_ua2[frame, :] - trial_ua1[frame, :]))
    ua_trial_z_frame = np.cross(ua_trial_y_frame, ua_trial_temp_frame) / np.linalg.norm(np.cross(ua_trial_y_frame, ua_trial_temp_frame))
    ua_trial_x_frame = np.cross(ua_trial_y_frame, ua_trial_z_frame) / np.linalg.norm(np.cross(ua_trial_y_frame, ua_trial_z_frame))
    #Right Upper Arm, tip facing ANTERIOR, O = ua1, +Y = superior, +X = anterior, +Z = lateral
    # ua_trial_y_frame = ((trial_ua3[frame, :] - trial_ua1[frame, :])) /(np.linalg.norm(trial_ua3[frame, :] - trial_ua1[frame, :]))
    # ua_trial_temp_frame = (trial_ua2[frame, :] - trial_ua1[frame, :]) / (np.linalg.norm(trial_ua2[frame, :] - trial_ua1[frame, :]))
    # ua_trial_z_frame = np.cross(ua_trial_temp_frame, ua_trial_y_frame) / np.linalg.norm(np.cross(ua_trial_temp_frame, ua_trial_y_frame))
    # ua_trial_x_frame = np.cross(ua_trial_y_frame, ua_trial_z_frame) / np.linalg.norm(np.cross(ua_trial_y_frame, ua_trial_z_frame))
    #Forearm, O = fa1, y = towards elbow, z = posterior, x = lateral
    fa_trial_y_frame = ((trial_fa3[frame, :] - trial_fa1[frame, :])) /(np.linalg.norm(trial_fa3[frame, :] - trial_fa1[frame, :]))
    fa_trial_temp_frame = (trial_fa2[frame,:] - trial_fa1[frame, :]) / (np.linalg.norm(trial_fa2[frame, :] - trial_fa1[frame, :]))
    fa_trial_x_frame = np.cross(fa_trial_y_frame, fa_trial_temp_frame) / np.linalg.norm(np.cross(fa_trial_y_frame, fa_trial_temp_frame))
    fa_trial_z_frame = np.cross(fa_trial_x_frame, fa_trial_y_frame) / np.linalg.norm(np.cross(fa_trial_x_frame, fa_trial_y_frame))
    #Hand, O = mcp2, Y = towards wrist, Z = radial X = dorsal
    hand_trial_y_frame = ((trial_rs[frame, :] - trial_mcp2[frame, :])) /(np.linalg.norm(trial_rs[frame, :] - trial_mcp2[frame, :]))
    hand_trial_temp_frame = (trial_us[frame,: ] - trial_mcp2[frame, :]) / (np.linalg.norm(trial_us[frame, :] - trial_mcp2[frame, :]))
    hand_trial_x_frame = np.cross(hand_trial_y_frame, hand_trial_temp_frame) / np.linalg.norm(np.cross(hand_trial_y_frame, hand_trial_temp_frame))
    hand_trial_z_frame = np.cross(hand_trial_x_frame, hand_trial_y_frame) / np.linalg.norm(np.cross(hand_trial_x_frame, hand_trial_y_frame))
      
# ******* 5. Define rotation matrix (global to local) for task clusters *******
    
# # set empty arrays for rotation matrix of each LCS
# hand_trial_GRL = np.zeros((3,3))
# fa_trial_GRL = np.zeros((3,3))    
# ua_trial_GRL = np.zeros((3,3))
# chest_trial_GRL = np.zeros((3,3))



# rotate each LCS to GCS
for frame in range(trial_frame_count):
    hand_trial_GRL = compute_GRL_rotation_matrix(hand_trial_x_frame, hand_trial_y_frame, hand_trial_z_frame)
    fa_trial_GRL = compute_GRL_rotation_matrix(fa_trial_x_frame, fa_trial_y_frame, fa_trial_z_frame)
    ua_trial_GRL = compute_GRL_rotation_matrix(ua_trial_x_frame, ua_trial_y_frame, ua_trial_z_frame)
    chest_trial_GRL = compute_GRL_rotation_matrix(chest_trial_x_frame, chest_trial_y_frame, chest_trial_z_frame)

# rotate individual markers to GCS

global_mcp2 = np.array([
    [np.dot(trial_mcp2[0], global_x), np.dot(trial_mcp2[1], global_y), np.dot(trial_mcp2[2], global_z)]])


# **** 6. Create virtual markers from the cal and task relationship
    # select one cluster per marker
# virtual = 1 cluster marker + (trial GRL*marker to cluster vector)
    
# Not recreating hand - no cluster 
    
# # Recreate forearm from FA1 - ME and LE, US and RS
#     #not using for recreation
# le_fa_trial_virtual = (trial_fa1 + np.dot(fa_trial_GRL, le_v_fa))
# me_fa_trial_virtual = (trial_fa1 + np.dot(fa_trial_GRL, me_v_fa))
# rs_fa_trial_virtual = (trial_fa1 + np.dot(fa_trial_GRL, rs_v_fa))
# us_fa_trial_virtual = (trial_fa1 + np.dot(fa_trial_GRL, us_v_fa))

#recreate upper arm from UA1 - ME, LE, R_ACR
le_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, le_v_ua))
me_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, me_v_ua))
racr_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, r_acr_v_ua))

#recreate torso from CHEST1 - SS XP 
ss_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, ss_v_chest))
xp_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, xp_v_chest))
c7_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, c7_v_chest))

# plot virtual markers and trial markers 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(le_ua_trial_virtual[0,0], le_ua_trial_virtual[0,1],le_ua_trial_virtual[0,2], c='r', marker='o', label='le')
# ax.scatter(me_ua_trial_virtual[0,0], me_ua_trial_virtual[0,1],me_ua_trial_virtual[0,2], c='r', marker='o', label='me')
# ax.scatter(racr_ua_trial_virtual[0,0], racr_ua_trial_virtual[0,1], racr_ua_trial_virtual[0,2], c='b', marker='o', label='racr')
# ax.scatter(ss_chest_trial_virtual[0,0], ss_chest_trial_virtual[0,1], ss_chest_trial_virtual[0,2], c='m', marker='o', label='ss')
# ax.scatter(xp_chest_trial_virtual[0,0], xp_chest_trial_virtual[0,1], xp_chest_trial_virtual[0,2], c='m', marker='o', label='xp')
# ax.scatter(c7_chest_trial_virtual[0,0], c7_chest_trial_virtual[0,1], c7_chest_trial_virtual[0,2], c='g', marker='o', label='c7')
# ax.scatter(trial_lacr[0,0], trial_lacr[0,1],trial_lacr[0,2], c='b', marker='o', label='lacr')
# ax.scatter(trial_rs[0,0], trial_rs[0,1],trial_rs[0,2], c='c', marker='o', label='rs')
# ax.scatter(trial_us[0,0], trial_us[0,1],trial_us[0,2], c='c', marker='o', label='us')
# ax.scatter(trial_mcp2[0,0], trial_mcp2[0,1],trial_mcp2[0,2], c='r', marker='o', label='mcp2')
# ax.scatter(trial_mcp5[0,0], trial_mcp5[0,1],trial_mcp5[0,2], c='r', marker='o', label='mcp5')
# ax.scatter(trial_fa1[0,0], trial_fa1[0,1],trial_fa1[0,2], c='k', marker='o', label='fa1')
# ax.scatter(trial_fa2[0,0], trial_fa2[0,1],trial_fa2[0,2], c='k', marker='o', label='fa2')
# ax.scatter(trial_fa3[0,0], trial_fa3[0,1],trial_fa3[0,2], c='k', marker='o', label='fa3')
# ax.scatter(trial_ua1[0,0], trial_ua1[0,1],trial_ua1[0,2], c='k', marker='o', label='ua1')
# ax.scatter(trial_ua2[0,0], trial_ua2[0,1],trial_ua2[0,2], c='k', marker='o', label='ua2')
# ax.scatter(trial_ua3[0,0], trial_ua3[0,1],trial_ua3[0,2], c='k', marker='o', label='ua3')

# ax.legend()

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # set lims to all positive, where they sit
# ax.set_xlim([0, 1300])
# ax.set_ylim([100, 1200])
# ax.set_zlim([0, 1300])
# # plt.show()

    
# butterworth filter 
def butter_low(signal: np.ndarray):
    """
    Apply a dual-pass 2nd order Butterworth low-pass filter to the input signal

  Inputs: 
    signal (numpy.ndarray): Input signal to be filtered

  Returns:
    numpy.ndarray: Low-pass filtered signals
    """
    fs = 100 # Sampling frequency
    order = 2 # filter order
    cutoff_f = 4 # Lowpass cutoff @ 4 Hz 
    Wn = (cutoff_f/(0.5*fs))
    sos = scipy.signal.butter(order, Wn, btype='lowpass', fs=fs, output='sos')
    lowpass_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    return lowpass_filtered_signal


# filter virtual markers and 'raw' hand markers

le_trial_filtered = np.empty_like(le_ua_trial_virtual)
me_trial_filtered = np.empty_like(me_ua_trial_virtual)
racr_trial_filtered = np.empty_like(racr_ua_trial_virtual)
ss_trial_filtered = np.empty_like(ss_chest_trial_virtual)
xp_trial_filtered = np.empty_like(xp_chest_trial_virtual)
c7_trial_filtered = np.empty_like(c7_chest_trial_virtual)
rs_trial_filtered = np.empty_like(trial_rs)
us_trial_filtered = np.empty_like(trial_us)
mcp2_trial_filtered = np.empty_like(trial_mcp2)
mcp5_trial_filtered = np.empty_like(trial_mcp5)


for i in range(le_ua_trial_virtual.shape[1]):
  le_trial_filtered[:,i] = butter_low(le_ua_trial_virtual[:, i])
  me_trial_filtered[:,i] = butter_low(me_ua_trial_virtual[:, i])
  racr_trial_filtered[:,i] = butter_low(racr_ua_trial_virtual[:,i])
  ss_trial_filtered[:,i] = butter_low(ss_chest_trial_virtual[:,i])
  xp_trial_filtered[:,i] = butter_low(xp_chest_trial_virtual[:,i])
  c7_trial_filtered[:,i] = butter_low(c7_chest_trial_virtual[:,i])
  rs_trial_filtered[:,i] = butter_low(trial_rs[:,i])
  us_trial_filtered[:,i] = butter_low(trial_us[:,i])
  mcp2_trial_filtered[:,i] = butter_low(trial_mcp2[:,i])
  mcp5_trial_filtered[:,i] = butter_low(trial_mcp5[:,i])


#Calculate joint centers
sjc_adjustment = np.array([0, -60, 0]) # -60 mm in the y
# set empty arrays to the size of marker trial data
wjc = np.empty_like(le_trial_filtered)
ejc = np.empty_like(le_trial_filtered)
sjc = np.empty_like(le_trial_filtered)
hand_origin = np.empty_like(le_trial_filtered)

for i in range(le_ua_trial_virtual.shape[1]):
  # wrist - midpoint between rs and us
  wjc[:,i] = (rs_trial_filtered[:,i] + us_trial_filtered[:,i])/2
  # elbow - midpoint between rs and us
  ejc[:,i] = (le_trial_filtered[:,i]) + (me_trial_filtered[:,i])/2
  # shoulder - acromion - 60mm in the y dir
  sjc[:,i] = (racr_trial_filtered[:,i] - sjc_adjustment[i])
  # hand origin
  hand_origin[:,i] = (mcp2_trial_filtered[:,i] + mcp5_trial_filtered[:,i])/2

#Create segment LCSs 
    # USE FILTERED DATA POINTS
# create empty lists to store segment LCS vectors
#forearm segment LCS vectors
fa_seg_trial_y = np.empty_like(le_trial_filtered)
fa_seg_trial_x = np.empty_like(le_trial_filtered)
fa_seg_trial_z = np.empty_like(le_trial_filtered)
fa_seg_trial_temp = np.empty_like(le_trial_filtered)

#upper arm / humerus segment LCS vectors
ua_seg_trial_y = np.empty_like(le_trial_filtered)
ua_seg_trial_z = np.empty_like(le_trial_filtered)
ua_seg_trial_x = np.empty_like(le_trial_filtered)
ua_seg_trial_temp = np.empty_like(le_trial_filtered)

#thorax segment LCS vectors
thrx_seg_trial_y = np.empty_like(le_trial_filtered)
thrx_seg_trial_z = np.empty_like(le_trial_filtered)
thrx_seg_trial_x = np.empty_like(le_trial_filtered)
thrx_seg_trial_temp = np.empty_like(le_trial_filtered)


#hand segment vectors
hand_seg_trial_x = np.empty_like(le_trial_filtered)
hand_seg_trial_y = np.empty_like(le_trial_filtered)
hand_seg_trial_z = np.empty_like(le_trial_filtered)
hand_seg_trial_temp = np.empty_like(le_trial_filtered)

#basis vectors
for i in range(le_ua_trial_virtual.shape[1]):
    #forearm
    fa_seg_trial_y[:,i] = ((ejc[:, i] - us_trial_filtered[:, i])) /(np.linalg.norm(ejc[:, i] - us_trial_filtered[:, i]))
    fa_seg_trial_temp[:,i] = (rs_trial_filtered[:,i] - us_trial_filtered[:,i]) / (np.linalg.norm(rs_trial_filtered[:,i] - us_trial_filtered[:,i]))
    fa_seg_trial_x = np.cross(fa_seg_trial_y, fa_seg_trial_temp) / np.linalg.norm(np.cross(fa_seg_trial_y, fa_seg_trial_temp))
    fa_seg_trial_z = np.cross(fa_seg_trial_x, fa_seg_trial_y) / np.linalg.norm(np.cross(fa_seg_trial_x, fa_seg_trial_y))

    #upper arm
    ua_seg_trial_y[:,i] = ((sjc[:,i] - ejc[:,i])) /(np.linalg.norm(sjc[:,i] - ejc[:,i]))
    ua_seg_trial_temp[:,i] = (ua_seg_trial_y[:,i] - fa_seg_trial_y[:,i]) / (np.linalg.norm(ua_seg_trial_y[:,i] - fa_seg_trial_y[:,i]))
    ua_seg_trial_z = np.cross(ua_seg_trial_y, ua_seg_trial_temp) / np.linalg.norm(np.cross(ua_seg_trial_y, ua_seg_trial_temp))
    ua_seg_trial_x = np.cross(ua_seg_trial_z, ua_seg_trial_y) / np.linalg.norm(np.cross(ua_seg_trial_z, ua_seg_trial_y))

    #thorax
    thrx_seg_trial_y = [0,1,0]
    thrx_seg_trial_temp[:,i] = (c7_trial_filtered[:,i] - ss_trial_filtered[:,i]) / (np.linalg.norm(c7_trial_filtered[:,i] - ss_trial_filtered[:,i]))
    thrx_seg_trial_z = np.cross(thrx_seg_trial_y, thrx_seg_trial_temp) / np.linalg.norm(np.cross(thrx_seg_trial_y, thrx_seg_trial_temp))
    thrx_seg_trial_x = np.cross(thrx_seg_trial_z, thrx_seg_trial_y) / np.linalg.norm(np.cross(thrx_seg_trial_z, thrx_seg_trial_y))

    hand_seg_o = hand_origin[:,i]
    hand_seg_trial_z[:,i] = (hand_origin[:,i] - mcp2_trial_filtered[:,i]) /(np.linalg.norm(hand_origin[:,i] - mcp2_trial_filtered[:,i]))
    hand_seg_trial_temp[:,i] = (rs_trial_filtered[:,i] - hand_origin[:,i]) / (np.linalg.norm(rs_trial_filtered[:,i] - hand_origin[:,i]))
    hand_seg_trial_x = np.cross(hand_seg_trial_temp, hand_seg_trial_z) / np.linalg.norm(np.cross(hand_seg_trial_temp,hand_seg_trial_z))
    hand_seg_trial_y = np.cross(hand_seg_trial_z, hand_seg_trial_x) / np.linalg.norm(np.cross(hand_seg_trial_z, hand_seg_trial_x))


# visual checks for segment LCS vectors

# # Vector Plotting 
    # forearm plot
# origin = [0, 0 ,0]
# ax.quiver(*origin, *fa_seg_trial_y[0], color='r', label='FA Y')
# ax.quiver(*origin, *fa_seg_trial_x[0], color='g', label='FA X')
# ax.quiver(*origin, *fa_seg_trial_z[0], color='b', label='FA Z')

# # Set plot limits
# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
# ax.set_zlim([-10, 10])
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

    # upper arm plot
# origin = [0, 0 ,0]
# ax.quiver(*origin, *ua_seg_trial_y, color='r', label='UA Y')
# ax.quiver(*origin, *ua_seg_trial_x, color='g', label='UA X')
# ax.quiver(*origin, *ua_seg_trial_z, color='b', label='UA Z')
#     #thorax plot
# origin = [0, 0 ,0]
# ax.quiver(*origin, *thrx_seg_trial_y, color='r', label='THORAX Y')
# ax.quiver(*origin, *thrx_seg_trial_x, color='g', label='THORAX X')
# ax.quiver(*origin, *thrx_seg_trial_z, color='b', label='THORAX Z')

# origin = hand_seg_o
# ax.quiver(*origin, *hand_seg_trial_y, color='r', label='FA Y')
# ax.quiver(*origin, *hand_seg_trial_x, color='g', label='FA X')
# ax.quiver(*origin, *hand_seg_trial_z, color='b', label='FA Z')

# CREATE DCMS FOR EACH SEGMENT
for i in range(le_ua_trial_virtual.shape[1]):
    hand_dcm_trial = np.stack((hand_seg_trial_x[:,i], hand_seg_trial_y[:,i], hand_seg_trial_z[:,i]), axis=-1)
    fa_dcm_trial = np.stack((fa_seg_trial_x[:,i], fa_seg_trial_y[:,i], fa_seg_trial_z[:,i]), axis=-1)
    ua_dcm_trial = np.stack((ua_seg_trial_x[:,i], ua_seg_trial_y[:,i], ua_seg_trial_z[:,i]), axis=-1)
    thrx_dcm_trial = np.stack((thrx_seg_trial_x[:,i], thrx_seg_trial_y[:,i], thrx_seg_trial_z[:,i]), axis=-1)
    #Define dcms between segments
    hand_fa_dcm_trial = hand_dcm_trial * fa_dcm_trial
    fa_ua_dcm_trial = fa_dcm_trial * ua_dcm_trial
    ua_thrx_dcm_trial = ua_dcm_trial * thrx_dcm_trial

# Calculate euler angles 

#forearm relative to hand (wrist) Z-X-Y (Wu 2005)
    # alpha (Y) = pronation (+) / supination (-) | displacement = proximal or distal translation
    # gamma (Z) = flexion (+) / extension (-) | displacement = radial / ulnar translation
    # beta (X) = radial (-) / ulnar deviation (+) | displacement - dorsal / volar translation

    #beta = asin (dcm (3,2)) # solve for beta first
    #alpha = acos (dcm(2,2)/cos(beta)) # 2,2 = cos(beta)cos(alpha)
    #gamma = acos (dcm (3,3)/cos(beta))
betaWrist = np.arcsin(hand_fa_dcm_trial[2,1])
alphaWrist = np.arccos(hand_fa_dcm_trial[1,1]/ np.cos(betaWrist))
gammaWrist = np.arccos(hand_fa_dcm_trial[2, 2] /np.cos(betaWrist))

wrist_alpha = np.degrees(alphaWrist)
wrist_beta = np.degrees(betaWrist)
wrist_gamma = np.degrees(gammaWrist)

#humerum relative to ulnar (elbow) Z-X-Y
    # alphaHF (Z) = flexion (+)/ hyperextension (-)  
    # gammaHF (Y) = axial rotation of the forearm | pronation (+) / supination (-) 
    # betaHF (X) = carrying angle, passive response to flex/ext, rarely reported

DCM_elbow = np.zeros((5622,3,3))
alpha_elbow = np.zeros((5622,1))
beta_elbow = np.zeros((5622,1))
gamma_elbow = np.zeros((5622,1))

for frame in range(le_ua_trial_virtual.shape[0]):
  #hand unit vectors
  x1_elbow = fa_seg_trial_x[frame,:]
  y1_elbow = fa_seg_trial_y[frame,:]
  z1_elbow = fa_seg_trial_z[frame,:]

  #forearm unit vectors
  x2_elbow = ua_seg_trial_x[frame,:]
  y2_elbow = ua_seg_trial_y[frame,:]
  z2_elbow = ua_seg_trial_z[frame,:]


  DCM_elbow[frame,:,:] = np.array([
      [np.dot(x1_elbow, x2_elbow), np.dot(x1_elbow, y2_elbow), np.dot(x1_elbow, z2_elbow)],
      [np.dot(y1_elbow, x2_elbow), np.dot(y1_elbow, y2_elbow), np.dot(y1_elbow, z2_elbow)],
      [np.dot(z1_elbow, x2_elbow), np.dot(z1_elbow, y2_elbow), np.dot(z1_elbow, z2_elbow)]])

  #row then col
  beta_elbow[frame,:] = np.arcsin(DCM_elbow[frame, 1,2])
  alpha_elbow[frame, :] = np.arccos((DCM_elbow[frame, 1,1])/np.cos(beta_elbow[frame]))
  gamma_elbow[frame,:] = np.arccos((DCM_elbow[frame, 2,2])/np.cos(beta_elbow[frame]))

alphadeg_elbow = np.degrees(alpha_elbow)
betadeg_elbow = np.degrees(beta_elbow)
gammadeg_elbow = np.degrees(gamma_elbow)
# cosB*cosA = -1.089
# cosA = -1.1/ cosB
# A = arccos(-1.1/cosB)
# sinB = -1.24
x = np.arange(5622)
plt.plot(x, gammadeg_elbow)
plt.show()



#thorax relative to upper arm (shoulder) Y-X-Y 
    #gammaH1 (Y) = plane of elevation | 0* = abduction / 90* = forward flexion
    #betaH (X) = elevation (-) 
    #gammaH2 (Y) = axial rotation | internal rot (+) / external rot (-)

#thorax relative to global cs Z-X-Y
    #alphaGT (Z) = flexion (-) / extension (+)
    #betaGT (X) = lateral flexion | right (+) / left (-)
    #gammaGT (Y) = axial rotation | left (+) / right (+)

