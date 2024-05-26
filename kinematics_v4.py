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


#'''IMPORT TRIAL FOLDER''' 
#update subject number
sub_num = "S01"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"

# sample rate
fs = 100

# IMPORT CALIBRATION TRIAL INTO DATA FRAME
# set cal file folder
cal_file = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_KinematicS/Digitized_TSV/d_S01_CAL.tsv"
# reads csv into a data frame
cal_raw = pd.read_csv(cal_file, sep='\t', header = 13)
# sets the row for the cal trial (note: frame#-1)
cal_frame = 5

# INDEX COLUMNS IN FILE TO LANDMARKS IN CAL AND TRIAL DFS
    #dictionary containing calibration individual markers
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

    #dictionary containing calibration individual markers
cal_clusters = {
    "fa1": cal_raw.iloc[cal_frame, 12:15].values,
    "fa2" : cal_raw.iloc[cal_frame, 15:18].values,
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

    # creating arrays containing marker indexing
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

cal_mcp2 = cal_raw.iloc[cal_frame, 0:3].values
cal_mcp5 = cal_raw.iloc[cal_frame,3:6].values
cal_rs = cal_raw.iloc[cal_frame,6:9].values
cal_us = cal_raw.iloc[cal_frame,9:12].values
cal_le = cal_raw.iloc[cal_frame, 21:24].values
cal_me = cal_raw.iloc[cal_frame, 24:27].values
cal_r_acr = cal_raw.iloc[cal_frame, 36:39].values
cal_ss = cal_raw.iloc[cal_frame, 42:45].values
cal_xp = cal_raw.iloc[cal_frame, 60:63].values
cal_c7 = cal_raw.iloc[cal_frame, 39:42].values
cal_l_acr = cal_raw.iloc[cal_frame, 63:66].values

# cal_ss = cal_raw.iloc[cal_frame, 42:45].values
# cal_xp = cal_raw.iloc[cal_frame, 60:63].values,
# cal_c7 = cal_raw.iloc[cal_frame, 39:42].values,


# ISB definition --- lab is already in ISB orientation
ISB_X = np.array([1, 0, 0])
ISB_Y = np.array([0, 1, 0])
ISB_Z = np.array([0, 0, 1])
ISB = np.array([ISB_X, ISB_Y, ISB_Z])

# If participant is facing door, then:
        # +X Local = -Z Global
        # +Y Local = +Y Global
        # +Z Local = +X Global
alt_CAL_ISB_X = np.array([0, 0, 1])
alt_CAL_ISB_Y = np.array([0, 1, 0])
alt_CAL_ISB_Z = np.array([-1, 0, 0])

alt_CAL_ISB = np.array([alt_CAL_ISB_X, alt_CAL_ISB_Y, alt_CAL_ISB_Z])

''' Did not rotate to ISB because lab xyz = isb xyz'''

# DEFINE LOCAL COORDINATE SYSTEMS OF CALIBRATION CLUSTERS #

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

#calculate relationship between global (identity matrix) and each LCS vector
def compute_GRL_rotation_matrix(LCS_v_x, LCS_v_y, LCS_v_z):
    """ Computes a Global to Local rotation matrix
    Inputs
    LCS_v_x = x vector norm of the LCS
    LCS_v_y = y vector norm of the LCS
    LCS_v_z = z vector norm of the LCS
    """
    trial_global_rot = np.zeros((3,3))
    global_x = np.array([1, 0, 0])
    global_y = np.array([0, 1, 0])
    global_z = np.array([0, 0, 1])

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

# DEFINE ROTATION MATRIX (GLOBAL TO LOCAL) FOR CAL CLUSTERS FROM BASIS VECTORS | grl = g to l
grl_cal_chest = compute_GRL_rotation_matrix(cal_chest_x, cal_chest_y, cal_chest_z)
grl_cal_ua = compute_GRL_rotation_matrix(cal_ua_x, cal_ua_y, cal_ua_z)
grl_cal_fa = compute_GRL_rotation_matrix(cal_fa_x, cal_fa_y, cal_fa_z)
grl_cal_hand = compute_GRL_rotation_matrix(cal_hand_x, cal_hand_y, cal_hand_z)

# VIRTUAL MARKERS | DEFINE RELATIONSHIP BETWEEN CLUSTERS AND MARKERS 
    # vector between marker and cl origin = (GRL rotation matrix) * (vector between markers))

# Chest to SS, XP, C7, chest origin set to chest5 (bottom right)
    # virtual marker = matrix multiplication between rotation matrix from GTL * (vector between desired marker and LCS origin) 
ss_chest5 = np.matmul(grl_cal_chest, (np.array(cal_ss) - np.array(cal_chest5)))
xp_chest5 = np.matmul(grl_cal_chest, (np.array(cal_xp) - np.array(cal_chest5)))
c7_chest5 = np.matmul(grl_cal_chest, (np.array(cal_c7) - np.array(cal_chest5)))

# Upper Arm to ME, LE, ua origin set to ua1
me_ua = np.matmul(grl_cal_ua, (np.array(cal_me) - np.array(cal_ua1)))
le_ua = np.matmul(grl_cal_ua, (np.array(cal_le) - np.array(cal_ua1)))
r_acr_ua = np.matmul(grl_cal_ua, (np.array(cal_r_acr) - np.array(cal_ua1)))

# Forearm to US, RS, fa origin set to fa1 
us_fa = np.matmul(grl_cal_fa, (np.array(cal_us) - np.array(cal_fa1)))
rs_fa = np.matmul(grl_cal_fa, (np.array(cal_rs) - np.array(cal_fa1)))
me_fa = np.matmul(grl_cal_ua, (np.array(cal_me) - np.array(cal_fa1)))
le_fa = np.matmul(grl_cal_ua, (np.array(cal_le) - np.array(cal_fa1)))



''' TASK TRIAL PROCESSING'''
# reads csv into a data frame
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV" 
trial_file = f"{trial_folder}/d_{sub_num}_EASY_LOW_7.tsv"
trial_raw = pd.read_csv(trial_file, sep='\t', header = 13) #sets csv to df

# DEFINE TRIAL CLUSTER AND MARKER INDEXING
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
trial_racr = trial_raw.iloc[:,36:39].values
trial_lacr = trial_raw.iloc[:,63:66].values
trial_chest1 = trial_raw.iloc[:,45:48].values
trial_chest2 = trial_raw.iloc[:,48:51].values
trial_chest3 = trial_raw.iloc[:,51:54].values
trial_chest4 = trial_raw.iloc[:,54:57].values
trial_chest5 = trial_raw.iloc[:,57:60].values
trial_ss = trial_raw.iloc[:,42:45].values
trial_xp = trial_raw.iloc[:,60:63].values
trial_c7 = trial_raw.iloc[:,39:42].values


# DEFINE LCS AND UNIT VECTORS FOR TASK CLUSTERS

# use imported task trial to iterate through cluster markers
# remaking coordinate systems using task clusters
trial_frame_count = len(trial_raw) # use this throughout to create array shapes and for loops

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
#creating LCS using the clusters | hand uses indiviudal markers
''' size = (length of trial,3)''' # CURRENTLY (3,)
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

# DEFINE ROTATION MATRIX (GLOBAL TO LOCAL) FOR TASK CLUSTER/SEGMENTS

hand_trial_GRL = compute_GRL_rotation_matrix(hand_trial_x_frame, hand_trial_y_frame, hand_trial_z_frame)
fa_trial_GRL = compute_GRL_rotation_matrix(fa_trial_x_frame, fa_trial_y_frame, fa_trial_z_frame)
ua_trial_GRL = compute_GRL_rotation_matrix(ua_trial_x_frame, ua_trial_y_frame, ua_trial_z_frame)
chest_trial_GRL = compute_GRL_rotation_matrix(chest_trial_x_frame, chest_trial_y_frame, chest_trial_z_frame)

''' should individual markers be rotated here?'''

# RECREATE MARKERS
    # not recreating mcp2, mcp5, rs or us 
#recreate upper arm from UA1 - ME, LE, R_ACR
le_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, le_ua))
me_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, me_ua))
racr_ua_trial_virtual = (trial_ua1 + np.dot(ua_trial_GRL, r_acr_ua))

#recreate torso from CHEST1 - SS XP 
ss_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, ss_chest5))
xp_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, xp_chest5))
c7_chest_trial_virtual = (trial_chest1 + np.dot(chest_trial_GRL, c7_chest5))

# CREATE BUTTERWORTH FILTER FUNCTION
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
    cutoff_f = 10 # Lowpass cutoff @ 10s Hz 
    Wn = (cutoff_f/(0.5*fs))
    sos = scipy.signal.butter(order, Wn, btype='lowpass', fs=fs, output='sos')
    lowpass_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    return lowpass_filtered_signal


#FILTER TRIAL MARKERS 
    #create empty arrays to store filtered marker data
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

 
for i in range(le_ua_trial_virtual.shape[1]): # iterate through frame length at x y z 
  le_trial_filtered[:,i] = butter_low(le_ua_trial_virtual[:, i])
  me_trial_filtered[:,i] = butter_low(me_ua_trial_virtual[:, i])
  racr_trial_filtered[:,i] = butter_low(racr_ua_trial_virtual[:,i])
  ss_trial_filtered[:,i] = butter_low(ss_chest_trial_virtual[:,i])
  xp_trial_filtered[:,i] = butter_low(xp_chest_trial_virtual[:,i])
  c7_trial_filtered[:,i] = butter_low(c7_chest_trial_virtual[:,i])
  rs_trial_filtered[:,i] = butter_low(trial_rs[:,i])
  us_trial_filtered[:,i] = butter_low(trial_us[:,i])
  mcp2_trial_filtered[:,i] = butter_low(trial_mcp2[:,i])
  mcp5_trial_filtered[:,i] = butter_low(trial_mcp5[:,i])\
  
#CALCULATE JOINT CENTERS
sjc_adjustment = np.array([0, -60, 0]) # -60 mm in the y
# set empty arrays to the size of marker trial data
wjc = np.empty_like(le_trial_filtered)
ejc = np.empty_like(le_trial_filtered)
sjc = np.empty_like(le_trial_filtered)
hand_origin = np.empty_like(le_trial_filtered)
hand_origin_dist = np.empty_like(le_trial_filtered)
hand_origin_prox = np.empty_like(le_trial_filtered)

for i in range(le_ua_trial_virtual.shape[1]):
  # wrist - midpoint between rs and us
  wjc[:,i] = (rs_trial_filtered[:,i] + us_trial_filtered[:,i])/2
  # elbow - midpoint between rs and us
  ejc[:,i] = (le_trial_filtered[:,i]) + (me_trial_filtered[:,i])/2
  # shoulder - acromion - 60mm in the y dir
  sjc[:,i] = (racr_trial_filtered[:,i] + sjc_adjustment[i])
  # hand origin - midpoint between mcps
  hand_origin_dist[:,i] = (mcp2_trial_filtered[:,i] + mcp5_trial_filtered[:,i])/2
  hand_origin_prox[:,i] = (rs_trial_filtered[:,i] + us_trial_filtered[:,i])/2
  hand_origin[:,i] = (hand_origin_dist[:,i] + hand_origin_prox[:,i])/2
#DEFINE TRIAL SEGMENT LCS USING FILTERED VIRTUAL MARKERS

# create empty lists to store trial segment xyz | size = (trial frames , xyz)
#forearm segment LCS vectors
fa_seg_trial_y = np.empty_like(le_trial_filtered)
fa_seg_trial_x = np.empty_like(le_trial_filtered)
fa_seg_trial_z = np.empty_like(le_trial_filtered)
fa_seg_trial_temp = np.empty_like(le_trial_filtered)

fa_seg_trial_y_norm  = np.empty_like(le_trial_filtered)
fa_seg_trial_x_norm  = np.empty_like(le_trial_filtered)
fa_seg_trial_z_norm = np.empty_like(le_trial_filtered)

#upper arm / humerus segment LCS vectors
ua_seg_trial_y = np.empty_like(le_trial_filtered)
ua_seg_trial_z = np.empty_like(le_trial_filtered)
ua_seg_trial_x = np.empty_like(le_trial_filtered)
ua_seg_trial_temp = np.empty_like(le_trial_filtered)

ua_seg_trial_y_norm  = np.empty_like(le_trial_filtered)
ua_seg_trial_x_norm  = np.empty_like(le_trial_filtered)
ua_seg_trial_z_norm = np.empty_like(le_trial_filtered)

#thorax segment LCS vectors
thrx_seg_trial_y = np.empty_like(le_trial_filtered)
thrx_seg_trial_z = np.empty_like(le_trial_filtered)
thrx_seg_trial_x = np.empty_like(le_trial_filtered)
thrx_seg_trial_temp = np.empty_like(le_trial_filtered)
thrx_y =np.empty_like(le_trial_filtered)

thrx_seg_trial_y_norm  = np.empty_like(le_trial_filtered)
thrx_seg_trial_x_norm  = np.empty_like(le_trial_filtered)
thrx_seg_trial_z_norm = np.empty_like(le_trial_filtered)

hand_seg_trial_x_norm = np.empty_like(le_trial_filtered)
hand_seg_trial_y_norm = np.empty_like(le_trial_filtered)
hand_seg_trial_z_norm = np.empty_like(le_trial_filtered)

#hand segment vectors
hand_seg_trial_x = np.empty_like(le_trial_filtered)
hand_seg_trial_y = np.empty_like(le_trial_filtered)
hand_seg_trial_z = np.empty_like(le_trial_filtered)
hand_seg_trial_temp = np.empty_like(le_trial_filtered)

hand_seg_trial_y_norm  = np.empty_like(le_trial_filtered)
hand_seg_trial_x_norm  = np.empty_like(le_trial_filtered)
hand_seg_trial_z_norm = np.empty_like(le_trial_filtered)

for frame in range(trial_frame_count):
#forearm
    fa_seg_trial_y[frame,:] = ((ejc[frame,:] - us_trial_filtered[frame,:])) 
    fa_seg_trial_temp[frame,:] = (rs_trial_filtered[frame,:] - us_trial_filtered[frame,:]) 
    fa_seg_trial_x[frame,:] = np.cross(fa_seg_trial_temp[frame,:], fa_seg_trial_y[frame,:])
    fa_seg_trial_z[frame,:] = np.cross(fa_seg_trial_x[frame,:], fa_seg_trial_y[frame,:]) 

    fa_seg_trial_y_norm[frame,:] = fa_seg_trial_y[frame,:]/(np.linalg.norm(fa_seg_trial_y[frame,:]))
    fa_seg_trial_x_norm[frame,:] = fa_seg_trial_x[frame,:]/(np.linalg.norm(fa_seg_trial_x[frame,:]))
    fa_seg_trial_z_norm[frame,:] = fa_seg_trial_z[frame,:]/(np.linalg.norm(fa_seg_trial_z[frame,:]))

    #upper arm
    ua_seg_trial_y[frame,:] = ((sjc[frame,:] - ejc[frame,:])) 
    ua_seg_trial_temp[frame,:] = (le_trial_filtered[frame,:] - ejc[frame,:]) 
    ua_seg_trial_x[frame,:] = np.cross(ua_seg_trial_y[frame,:], ua_seg_trial_temp[frame,:]) 
    ua_seg_trial_z[frame,:] = np.cross(ua_seg_trial_x[frame,:], ua_seg_trial_y[frame,:]) 
    ua_seg_trial_x[frame,:] = np.cross(ua_seg_trial_y[frame,:], ua_seg_trial_temp[frame,:]) 
    ua_seg_trial_z[frame,:] = np.cross(ua_seg_trial_x[frame,:], ua_seg_trial_y[frame,:]) 

    ua_seg_trial_y_norm[frame,:] = ua_seg_trial_y[frame,:] / (np.linalg.norm(ua_seg_trial_y[frame,:]))
    ua_seg_trial_x_norm[frame,:] = ua_seg_trial_x[frame,:] / (np.linalg.norm(ua_seg_trial_x[frame,:]))
    ua_seg_trial_z_norm[frame,:] = ua_seg_trial_z[frame,:] / (np.linalg.norm(ua_seg_trial_z[frame,:]))
    
    #thorax, origin set to ss
    thrx_y = np.tile([0, 1, 0], (trial_frame_count, 1)) #np.tile creates an array with the repeating number, rows = frames
    
    thrx_seg_trial_y[frame,:] = (thrx_y[frame,:] - ss_trial_filtered[frame, :])
    thrx_seg_trial_temp[frame,:] = (c7_trial_filtered[frame, :] - ss_trial_filtered[frame, :])
    thrx_seg_trial_z[frame,:] = np.cross(thrx_seg_trial_y[frame, :], thrx_seg_trial_temp[frame, :]) 
    thrx_seg_trial_x[frame,:] = np.cross(thrx_seg_trial_z[frame, :], thrx_seg_trial_y[frame, :])
    # print(fa_seg_trial_x_norm.s
    #   thorax
    # thrx_seg_trial_y = [0,1,0] 
    # thrx_seg_trial_temp = (c7_trial_filtered - ss_trial_filtered) / (np.linalg.norm(c7_trial_filtered - ss_trial_filtered))
    # thrx_seg_trial_z = np.cross(thrx_seg_trial_y, thrx_seg_trial_temp) / np.linalg.norm(np.cross(thrx_seg_trial_y, thrx_seg_trial_temp))
    # thrx_seg_trial_x = np.cross(thrx_seg_trial_z, thrx_seg_trial_y) / np.linalg.norm(np.cross(thrx_seg_trial_z, thrx_seg_trial_y))

    thrx_seg_trial_y_norm[frame,:] = thrx_seg_trial_y[frame,:] / (np.linalg.norm(thrx_seg_trial_y[frame,:]))
    thrx_seg_trial_x_norm[frame,:] = thrx_seg_trial_x[frame,:] / (np.linalg.norm(thrx_seg_trial_x[frame,:]))
    thrx_seg_trial_z_norm[frame,:] = thrx_seg_trial_z[frame,:] / (np.linalg.norm(thrx_seg_trial_z[frame,:]))
    
    # thrx_seg_trial_y_norm[frame,:] = thrx_seg_trial_y[frame,:] / (np.linalg.norm(thrx_seg_trial_y[frame,:]))
    # thrx_seg_trial_x_norm[frame,:] = thrx_seg_trial_x[frame,:] / (np.linalg.norm(thrx_seg_trial_x[frame,:]))
    # thrx_seg_trial_z_norm[frame,:] = thrx_seg_trial_z[frame,:] / (np.linalg.norm(thrx_seg_trial_z[frame,:]))

    hand_seg_trial_y[frame,:] = (hand_origin_prox[frame,:] - hand_origin[frame,:])
    hand_seg_trial_temp[frame,:] = (mcp2_trial_filtered[frame,:] - hand_origin[frame,:])
    hand_seg_trial_x[frame,:] = np.cross(hand_seg_trial_temp[frame,:], hand_seg_trial_y[frame,:]) 
    hand_seg_trial_z[frame,:] = np.cross(hand_seg_trial_x[frame,:], hand_seg_trial_y[frame,:]) 

    hand_seg_trial_y_norm[frame,:] = hand_seg_trial_y[frame,:] / (np.linalg.norm(hand_seg_trial_y[frame,:]))
    hand_seg_trial_x_norm[frame,:] = hand_seg_trial_x[frame,:] / (np.linalg.norm(hand_seg_trial_x[frame,:]))
    hand_seg_trial_z_norm[frame,:] = hand_seg_trial_z[frame,:] / (np.linalg.norm(hand_seg_trial_z[frame,:]))


# Visual checks for anatomical lcs

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


# # DEFINE SEGMENT DIRECTION COSINE MATRICES

# ''' WRIST'''
#forearm relative to hand (wrist) Z-X-Y (Wu 2005)
    # alpha (Y) = pronation (+) / supination (-) | displacement = proximal or distal translation
    # gamma (Z) = flexion (+) / extension (-) | displacement = radial / ulnar translation
    # beta (X) = radial (-) / ulnar deviation (+) | displacement - dorsal / volar translation

    #beta = asin (dcm (3,2)) # solve for beta first
    #alpha = acos (dcm(2,2)/cos(beta)) # 2,2 = cos(beta)cos(alpha)
    #gamma = acos (dcm (3,3)/cos(beta))
DCM_wrist = np.zeros((3,3))
alpha_wrist = np.zeros((trial_frame_count,1))
beta_wrist = np.zeros((trial_frame_count,1))
gamma_wrist = np.zeros((trial_frame_count,1))
# compute ijk from x y z basis vectors of each segment
for frame in range(trial_frame_count):
  hand_seg_i = [hand_seg_trial_x_norm[frame,0], hand_seg_trial_y_norm[frame,0], hand_seg_trial_z_norm[frame,0]]
  hand_seg_j = [hand_seg_trial_x_norm[frame,1], hand_seg_trial_y_norm[frame,1], hand_seg_trial_z_norm[frame,1]]
  hand_seg_k = [hand_seg_trial_x_norm[frame,2], hand_seg_trial_y_norm[frame,2], hand_seg_trial_z_norm[frame,2]]
  hand_dcm_ijk = np.vstack((hand_seg_i, hand_seg_j,hand_seg_k))
  hand_dcm_ijk_t = np.transpose(hand_dcm_ijk) #transposed dcm


#   hand_seg_i = [hand_seg_trial_x_norm[frame,0], hand_seg_trial_x_norm[frame,1], hand_seg_trial_x_norm[frame,2]]
#   hand_seg_j = [hand_seg_trial_y_norm[frame,0], hand_seg_trial_y_norm[frame,1], hand_seg_trial_y_norm[frame,2]]
#   hand_seg_k = [hand_seg_trial_z_norm[frame,0], hand_seg_trial_z_norm[frame,1], hand_seg_trial_z_norm[frame,2]]
#   hand_dcm_ijk = np.vstack((hand_seg_i, hand_seg_j,hand_seg_k))
#   hand_dcm_ijk_t = np.transpose(hand_dcm_ijk) #transposed dcm
  # print(hand_dcm_ijk)
  fa_seg_i = [fa_seg_trial_x_norm[frame,0], fa_seg_trial_y_norm[frame,0], fa_seg_trial_z_norm[frame,0]]
  fa_seg_j = [fa_seg_trial_x_norm[frame,1], fa_seg_trial_y_norm[frame,1], fa_seg_trial_z_norm[frame,1]]
  fa_seg_k = [fa_seg_trial_x_norm[frame,2], fa_seg_trial_y_norm[frame,2], fa_seg_trial_z_norm[frame,2]]
  fa_dcm_ijk = np.vstack((fa_seg_i, fa_seg_j, fa_seg_k))
  fa_dcm_ijk_t = np.transpose(fa_dcm_ijk) #transposed dcm

#   fa_seg_i = [fa_seg_trial_x_norm[frame,0], fa_seg_trial_x_norm[frame,0], fa_seg_trial_x_norm[frame,0]]
#   fa_seg_j = [fa_seg_trial_y_norm[frame,0], fa_seg_trial_y_norm[frame,1], fa_seg_trial_z_norm[frame,1]]
#   fa_seg_k = [fa_seg_trial_x_norm[frame,2], fa_seg_trial_y_norm[frame,2], fa_seg_trial_z_norm[frame,2]]
#   fa_dcm_ijk = np.vstack((fa_seg_i, fa_seg_j, fa_seg_k))
  fa_dcm_ijk_t = np.transpose(fa_dcm_ijk) #transposed dcm
  # distal segment should be transposed
  wrist_dcm = np.matmul(fa_dcm_ijk, hand_dcm_ijk_t)
  #iterate through angles at each frame
  beta_wrist[frame,:] = np.arcsin(wrist_dcm[1,2])
  alpha_wrist[frame, :] = np.arccos((wrist_dcm[1,1])/np.cos(beta_wrist[frame]))
  gamma_wrist[frame,:] = np.arccos((wrist_dcm[2,2])/np.cos(beta_wrist[frame]))
#convert angles to degrees
alphadeg_wrist = np.degrees(alpha_wrist)
betadeg_wrist = np.degrees(beta_wrist)
gammadeg_wrist = np.degrees(gamma_wrist)

x = np.linspace(0,trial_frame_count,trial_frame_count)
# plt.set_xlim(0.55, 0.56)
plt.plot(x, gammadeg_wrist)
plt.title('wrist angle gamma in degrees')
plt.show()
# x = np.linspace(0,trial_frame_count,trial_frame_count)
# # plt.set_xlim(0.55, 0.56)
# plt.plot(x, gammadeg_wrist)
# plt.show()

# ''' ELBOW '''
#humerum relative to ulnar (elbow) Z-X-Y
    # alphaHF (Z) = flexion (+)/ hyperextension (-)  
    # gammaHF (Y) = axial rotation of the forearm | pronation (+) / supination (-) 
    # betaHF (X) = carrying angle, passive response to flex/ext, rarely reported
np.set_printoptions(precision = 5)
DCM_elbow = np.zeros((trial_frame_count,3,3))
alpha_elbow = np.zeros((trial_frame_count,1))
beta_elbow = np.zeros((trial_frame_count,1))
gamma_elbow = np.zeros((trial_frame_count,1))

print(trial_frame_count)
for frame in range(trial_frame_count):
#   fa_seg_i = [fa_seg_trial_x_norm[frame,0], fa_seg_trial_y_norm[frame,0], fa_seg_trial_z_norm[frame,0]]
#   fa_seg_j = [fa_seg_trial_x_norm[frame,1], fa_seg_trial_y_norm[frame,1], fa_seg_trial_z_norm[frame,1]]
#   fa_seg_k = [fa_seg_trial_x_norm[frame,2], fa_seg_trial_y_norm[frame,2], fa_seg_trial_z_norm[frame,2]]
  
#   fa_dcm_ijk = np.vstack((fa_seg_i, fa_seg_j, fa_seg_k))
#   fa_dcm_ijk_t = np.transpose(fa_dcm_ijk)
#   #print(fa_dcm_ijk)
#   ua_seg_i = [ua_seg_trial_x_norm[frame,0], ua_seg_trial_y_norm[frame,0], ua_seg_trial_z_norm[frame,0]]
#   ua_seg_j = [ua_seg_trial_x_norm[frame,1], ua_seg_trial_y_norm[frame,1], ua_seg_trial_z_norm[frame,1]]
#   ua_seg_k = [ua_seg_trial_x_norm[frame,2], ua_seg_trial_y_norm[frame,2], ua_seg_trial_z_norm[frame,2]]
#   ua_dcm_ijk = np.vstack((ua_seg_i, ua_seg_j, ua_seg_k))
#   #print(ua_dcm_ijk)

  
#   fa_seg_i = [fa_seg_trial_x_norm[frame,0], fa_seg_trial_x_norm[frame,1], fa_seg_trial_x_norm[frame,2]]
#   fa_seg_j = [fa_seg_trial_y_norm[frame,0], fa_seg_trial_y_norm[frame,1], fa_seg_trial_y_norm[frame,2]]
#   fa_seg_k = [fa_seg_trial_z_norm[frame,0], fa_seg_trial_z_norm[frame,1], fa_seg_trial_z_norm[frame,2]]
  
#   fa_dcm_ijk = np.vstack((fa_seg_i, fa_seg_j, fa_seg_k))
#   fa_dcm_ijk_t = np.transpose(fa_dcm_ijk)
#   #print(fa_dcm_ijk)
#   ua_seg_i = [ua_seg_trial_x_norm[frame,0], ua_seg_trial_x_norm[frame,1], ua_seg_trial_x_norm[frame,2]]
#   ua_seg_j = [ua_seg_trial_y_norm[frame,0], ua_seg_trial_y_norm[frame,1], ua_seg_trial_y_norm[frame,2]]
#   ua_seg_k = [ua_seg_trial_z_norm[frame,0], ua_seg_trial_z_norm[frame,1], ua_seg_trial_z_norm[frame,2]]
#   ua_dcm_ijk = np.vstack((ua_seg_i, ua_seg_j, ua_seg_k))

    elbow_dcm = np.array( 
        [
        [ 
                np.dot(fa_seg_trial_x_norm[frame], ua_seg_trial_x_norm[frame]),
                np.dot(fa_seg_trial_x_norm[frame], ua_seg_trial_y_norm[frame]),
                np.dot(fa_seg_trial_x_norm[frame], ua_seg_trial_x_norm[frame])
            ],
            [
                np.dot(fa_seg_trial_y_norm[frame], ua_seg_trial_x_norm[frame]),
                np.dot(fa_seg_trial_y_norm[frame], ua_seg_trial_y_norm[frame]),
                np.dot(fa_seg_trial_y_norm[frame], ua_seg_trial_z_norm[frame]),
            ],
            [
                np.dot(fa_seg_trial_z_norm[frame], ua_seg_trial_x_norm[frame]),
                np.dot(fa_seg_trial_z_norm[frame], ua_seg_trial_y_norm[frame]),
                np.dot(fa_seg_trial_z_norm[frame], ua_seg_trial_z_norm[frame]),
            ] 
        ]
    )
    #elbow_dcm = np.dot(ua_dcm_ijk, fa_dcm_ijk_t)
    #   #row then col
    beta_elbow[frame,:] = np.arcsin(elbow_dcm[1,2])
    alpha_elbow[frame, :] = np.arccos((elbow_dcm[1,1])/np.cos(beta_elbow[frame]))
    gamma_elbow[frame,:] = np.arccos((elbow_dcm[2,2])/np.cos(beta_elbow[frame]))

# print(beta_elbow)
    alphadeg_elbow = np.degrees(alpha_elbow)
    betadeg_elbow = np.degrees(beta_elbow)
    gammadeg_elbow = np.degrees(gamma_elbow)


# x = np.linspace(0,trial_frame_count,trial_frame_count)
# # plt.set_xlim(0.55, 0.56)
# plt.plot(x, gammadeg_elbow)
# plt.title('elbow angle galpha in degrees')
# plt.show()
x = np.linspace(0,trial_frame_count,trial_frame_count)
# plt.set_xlim(0.55, 0.56)
plt.plot(x, alphadeg_elbow)
plt.show()

''' SHOULDER '''

# DCM_ghj = np.zeros((frame_count,3,3))
# beta_ghj = np.zeros((frame_count,1))
# gamma1_ghj = np.zeros((frame_count,1))
# gamma2_ghj = np.zeros((frame_count,1))


# for frame in range(frame_count):
#   #ua/humerus unit vectors
#   x1_ghj = ua_seg_trial_x[frame,:]
#   y1_ghj = ua_seg_trial_y[frame,:]
#   z1_ghj = ua_seg_trial_z[frame,:]

#   #thorax unit vectors
#   x2_ghj = thrx_seg_trial_x[frame,:]
#   y2_ghj = thrx_seg_trial_y[frame,:]
#   z2_ghj = thrx_seg_trial_z[frame,:]


#   DCM_ghj[frame,:,:] = np.array([
#       [np.dot(x1_ghj, x2_ghj), np.dot(x1_ghj, y2_ghj), np.dot(x1_ghj, z2_ghj)],
#       [np.dot(y1_ghj, x2_ghj), np.dot(y1_ghj, y2_ghj), np.dot(y1_ghj, z2_ghj)],
#       [np.dot(z1_ghj, x2_ghj), np.dot(z1_ghj, y2_ghj), np.dot(z1_ghj, z2_ghj)]])

#   #row then col
#   beta_ghj[frame,:] = np.arccos((DCM_ghj[frame, 1,1]))
#   gamma1_ghj[frame, :] = np.arcsin((DCM_ghj[frame, 0,1])/np.sin(beta_ghj[frame, 0]))
#   gamma2_ghj[frame,:] = np.arcsin((DCM_ghj[frame, 1,0])/np.sin(beta_ghj[frame,0]))

# betadeg_ghj = np.degrees(beta_ghj)
# gamma1deg_ghj = np.degrees(gamma1_ghj)
# gamma2deg_ghj = np.degrees(gamma2_ghj)