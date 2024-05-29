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
from enum import IntEnum
from typing import Dict, Tuple
from numpy import typing as npt


#'''IMPORT TRIAL FOLDER''' 
#update subject number
sub_num = "S02"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"

# sample rate
fs = 100

# IMPORT CALIBRATION TRIAL INTO DATA FRAME
# set cal file folder
cal_file = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_KinematicS/Digitized_TSV/d_S02_CAL.tsv"
# reads csv into a data frame
cal_raw = pd.read_csv(cal_file, sep='\t', header = 11)
# sets the row for the cal trial (note: frame#-1)
cal_frame = 5

# INDEX COLUMNS IN FILE TO LANDMARKS IN CAL AND TRIAL DFS
    #dictionary containing calibration individual markers
cal_markers = {
    "mcp2": cal_raw[["MCP2 X", "MCP2 Y", "MCP2 Z"]].values,
    "mcp5": cal_raw[["MCP5 X", "MCP5 Y", "MCP5 Z"]].values,
    "rs": cal_raw[["RS X", "RS Y", "RS Z"]].values,
    "us": cal_raw[["US X", "US Y", "US Z"]].values,
    "le": cal_raw[["LE X", "LE Y", "LE Z"]].values,
    "me": cal_raw[["ME X", "ME Y", "ME Z"]].values,
    "r_acr": cal_raw[["R_ACR X", "R_ACR Y", "R_ACR Z"]].values,
    "ss": cal_raw[["SS X", "SS Y", "SS Z"]].values,
    "xp": cal_raw[["XP X", "XP Y", "XP Z"]].values,
    "c7": cal_raw[["C7 X", "C7 Y", "C7 Z"]].values,
    "l_acr": cal_raw[["L_ACR X", "L_ACR Y", "L_ACR Z"]].values,
}
for marker in cal_markers:
   cal_markers[marker] = cal_markers[marker][cal_frame]

    #dictionary containing calibration individual markers
cal_clusters = {
    "fa1": cal_raw[["FA1 X", "FA1 Y", "FA1 Z"]].values,
    "fa2": cal_raw[["FA2 X", "FA2 Y", "FA2 Z"]].values,
    "fa3": cal_raw[["FA3 X", "FA3 Y", "FA3 Z"]].values,
    "ua1": cal_raw[["UA1 X", "UA1 Y", "UA1 Z"]].values,
    "ua2": cal_raw[["UA2 X", "UA2 Y", "UA2 Z"]].values,
    "ua3": cal_raw[["UA3 X", "UA3 Y", "UA3 Z"]].values,
    "chest1": cal_raw[["CHEST1 X", "CHEST1 Y", "CHEST1 Z"]].values,
    "chest2": cal_raw[["CHEST2 X", "CHEST2 Y", "CHEST2 Z"]].values,
    "chest3": cal_raw[["CHEST3 X", "CHEST3 Y", "CHEST3 Z"]].values,
    "chest4": cal_raw[["CHEST4 X", "CHEST4 Y", "CHEST4 Z"]].values,
    "chest5": cal_raw[["CHEST5 X", "CHEST5 Y", "CHEST5 Z"]].values,
}

for marker in cal_clusters:
   cal_clusters[marker] = cal_clusters[marker][cal_frame]
    # creating arrays containing marker indexing
# cal cluster marker arrays
cal_fa1 = cal_clusters['fa1']
cal_fa2 = cal_clusters['fa2']
cal_fa3 = cal_clusters['fa3']
cal_ua1 = cal_clusters['ua1']
cal_ua2 = cal_clusters['ua2']
cal_ua3 = cal_clusters['ua3']
cal_chest1 = cal_clusters['chest1']
cal_chest2 = cal_clusters['chest2']
cal_chest3 = cal_clusters['chest3']
cal_chest4 = cal_clusters['chest4']
cal_chest5 = cal_clusters['chest5']

cal_mcp2 = cal_markers['mcp2']
cal_mcp5 = cal_markers['mcp5']
cal_rs = cal_markers['rs']
cal_us = cal_markers['us']
cal_le =  cal_markers['le']
cal_me =  cal_markers['me']
cal_r_acr = cal_markers['r_acr']
cal_ss =  cal_markers['ss']
cal_xp =  cal_markers['xp']
cal_c7 =  cal_markers['c7']
cal_l_acr = cal_markers['l_acr']


# ISB definition --- lab is already in ISB orientation
ISB_X = np.array([1, 0, 0])
ISB_Y = np.array([0, 1, 0])
ISB_Z = np.array([0, 0, 1])
ISB = np.array([ISB_X, ISB_Y, ISB_Z])

class Axis(IntEnum):
    """Enum for the axes of the coordinate system."""

    X = 0
    Y = 1
    Z = 2

def _plot_marker(
    fig: plt.Figure, marker: npt.NDArray, color: str = "black", name: str = None
) -> None:
    """Plot a marker on a 3D plot.

    Args:
        fig (plt.Figure): The matplotlib figure object on which to plot the marker.
        marker (npt.NDArray): The coordinates of the marker in the form of a NumPy
            array with shape (3,).
        color (str, optional): The color of the marker. Defaults to "black".
        name (str, optional): The name of the marker. Defaults to None.
    """
    fig.scatter(marker[0], marker[1], marker[2], color=color)
    if name:
        fig.text(marker[0], marker[1], marker[2], name)


def _plot_coordinate_system(
    fig: plt.Figure, origin: npt.NDArray, unit_vectors: npt.NDArray
) -> None:
    """Plot the Local Coordinate System on a 3D plot.

    Args:
        fig (plt.Figure): The matplotlib figure object to plot on.
        origin (npt.NDArray): The origin of the coordinate system.
        unit_vectors (npt.NDArray): The unit vectors of the coordinate system.
    """
    fig.quiver(
        origin[0],
        origin[1],
        origin[2],
        unit_vectors[Axis.X][0],
        unit_vectors[Axis.X][1],
        unit_vectors[Axis.X][2],
        arrow_length_ratio=0.1,
        length=10.0,
        color="r",
    )
    fig.quiver(
        origin[0],
        origin[1],
        origin[2],
        unit_vectors[Axis.Y][0],
        unit_vectors[Axis.Y][1],
        unit_vectors[Axis.Y][2],
        arrow_length_ratio=0.1,
        length=10.0,
        color="g",
    )
    fig.quiver(
        origin[0],
        origin[1],
        origin[2],
        unit_vectors[Axis.Z][0],
        unit_vectors[Axis.Z][1],
        unit_vectors[Axis.Z][2],
        arrow_length_ratio=0.1,
        length=10.0,
        color="b",
    )



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

# VISUAL CHECK FOR NEUTRAL POSTURE
# USE THIS DCM SET UP
    # Set forearm pre-dcm to the unit vectors from the anatomical lcs
fa_cal_dcm_ijk = np.vstack((cal_fa_x, cal_fa_y, cal_fa_z))
fa_cal_dcm_ijk_t = np.transpose(fa_cal_dcm_ijk)
    # Set Uppe Arm pre-dcm to the unit vectors from the anatomical lcs
hand_cal_dcm_ijk = np.vstack((cal_hand_x, cal_hand_y, cal_hand_z))
hand_cal_dcm_ijk_t = np.transpose(hand_cal_dcm_ijk)

ua_cal_dcm_ijk = np.vstack((cal_ua_x, cal_ua_y, cal_ua_z))
ua_cal_dcm_ijk_t = np.transpose(ua_cal_dcm_ijk)

thrx_cal_dcm_ijk = np.vstack((cal_chest_x, cal_chest_y, cal_chest_z))
thrx_cal_dcm_ijk = np.transpose(thrx_cal_dcm_ijk)
# dot product between the two anatomical LCS 
cal_wrist_dcm = np.array( 
    [
    [ 
            np.dot(hand_cal_dcm_ijk_t[0], fa_cal_dcm_ijk[0]),
            np.dot(hand_cal_dcm_ijk_t[0], fa_cal_dcm_ijk[1]),
            np.dot(hand_cal_dcm_ijk_t[0], fa_cal_dcm_ijk[2])
        ],
        [
            np.dot(hand_cal_dcm_ijk_t[1], fa_cal_dcm_ijk[0]),
            np.dot(hand_cal_dcm_ijk_t[1], fa_cal_dcm_ijk[1]),
            np.dot(hand_cal_dcm_ijk_t[1], fa_cal_dcm_ijk[2]),
        ],
        [
            np.dot(hand_cal_dcm_ijk_t[2], fa_cal_dcm_ijk[0]),
            np.dot(hand_cal_dcm_ijk_t[2], fa_cal_dcm_ijk[1]),
            np.dot(hand_cal_dcm_ijk_t[2], fa_cal_dcm_ijk[2]),
        ] 
    ]
)
cal_elbow_dcm = np.array( 
    [
    [ 
            np.dot(fa_cal_dcm_ijk_t[0], ua_cal_dcm_ijk[0]),
            np.dot(fa_cal_dcm_ijk_t[0], ua_cal_dcm_ijk[1]),
            np.dot(fa_cal_dcm_ijk_t[0], ua_cal_dcm_ijk[2])
        ],
        [
            np.dot(fa_cal_dcm_ijk_t[1], ua_cal_dcm_ijk[0]),
            np.dot(fa_cal_dcm_ijk_t[1], ua_cal_dcm_ijk[1]),
            np.dot(fa_cal_dcm_ijk_t[1], ua_cal_dcm_ijk[2]),
        ],
        [
            np.dot(fa_cal_dcm_ijk_t[2], ua_cal_dcm_ijk[0]),
            np.dot(fa_cal_dcm_ijk_t[2], ua_cal_dcm_ijk[1]),
            np.dot(fa_cal_dcm_ijk_t[2], ua_cal_dcm_ijk[2]),
        ] 
    ]
)
cal_shoulder_dcm = np.array( 
    [
    [ 
            np.dot(ua_cal_dcm_ijk_t[0], thrx_cal_dcm_ijk[0]),
            np.dot(ua_cal_dcm_ijk_t[0], thrx_cal_dcm_ijk[1]),
            np.dot(ua_cal_dcm_ijk_t[0], thrx_cal_dcm_ijk[2])
        ],
        [
            np.dot(ua_cal_dcm_ijk_t[1], thrx_cal_dcm_ijk[0]),
            np.dot(ua_cal_dcm_ijk_t[1], thrx_cal_dcm_ijk[1]),
            np.dot(ua_cal_dcm_ijk_t[1], thrx_cal_dcm_ijk[2]),
        ],
        [
            np.dot(ua_cal_dcm_ijk_t[2], thrx_cal_dcm_ijk[0]),
            np.dot(ua_cal_dcm_ijk_t[2], thrx_cal_dcm_ijk[1]),
            np.dot(ua_cal_dcm_ijk_t[2], thrx_cal_dcm_ijk[2]),
        ] 
    ]
)

#   #row then col
cal_beta_wrist = np.arcsin(cal_wrist_dcm[1,2])
cal_alpha_wrist = np.arccos((cal_wrist_dcm[1,1])/np.cos(cal_beta_wrist))
cal_gamma_wrist = np.arccos((cal_wrist_dcm[2,2])/np.cos(cal_beta_wrist))

# print(beta_wrist)
cal_alphadeg_wrist = np.degrees(cal_alpha_wrist)
cal_betadeg_wrist = np.degrees(cal_beta_wrist)
cal_gammadeg_wrist = np.degrees(cal_gamma_wrist)

print(f'Calibration trial alpha wrist angle {cal_alphadeg_wrist}')
print(f'Calibration trial beta wrist angle {cal_betadeg_wrist}')
print(f'Calibration trial gamma wrist angle {cal_gammadeg_wrist}')

cal_beta_elbow = np.arcsin(cal_elbow_dcm[1,2])
cal_alpha_elbow = np.arccos((cal_elbow_dcm[1,1])/np.cos(cal_beta_elbow))
cal_gamma_elbow = np.arccos((cal_elbow_dcm[2,2])/np.cos(cal_beta_elbow))

# print(beta_elbow)
cal_alphadeg_elbow = np.degrees(cal_alpha_elbow)
cal_betadeg_elbow = np.degrees(cal_beta_elbow)
cal_gammadeg_elbow = np.degrees(cal_gamma_elbow)

print(f'Calibration trial alpha elbow angle {cal_alphadeg_elbow}')
print(f'Calibration trial beta elbow angle {cal_betadeg_elbow}')
print(f'Calibration trial gamma elbow angle {cal_gammadeg_elbow}')

cal_beta_shoulder = np.arcsin(cal_shoulder_dcm[1,2])
cal_gamma1_shoulder = np.arccos((cal_shoulder_dcm[1,1])/np.cos(cal_beta_shoulder))
cal_gamma2_shoulder = np.arccos((cal_shoulder_dcm[2,2])/np.cos(cal_beta_shoulder))

# print(beta_shoulder)
cal_betadeg_shoulder = np.degrees(cal_beta_shoulder)
cal_gamma1deg_shoulder = np.degrees(cal_gamma1_shoulder)
cal_gamma2deg_shoulder = np.degrees(cal_gamma2_shoulder)

print(f'Calibration trial beta shoulder angle {cal_betadeg_shoulder}')
print(f'Calibration trial gamma1 shoulder angle {cal_gamma1deg_shoulder}')
print(f'Calibration trial gamma2 shoulder angle {cal_gamma2deg_shoulder}')

# PLOT CAL MARKERS
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_xlim(0,2000)
# ax.set_ylim(0,2000)
# ax.set_zlim(0,2000)
# _plot_marker(ax, cal_chest5, 'blue', 'chest5')
# _plot_marker(ax, cal_chest4, 'blue', 'chest4')
# _plot_marker(ax, cal_chest3, 'blue', 'chest3')
# _plot_marker(ax, cal_chest2, 'blue', 'chest2')
# _plot_marker(ax, cal_chest1, 'blue', 'chest1')
# _plot_marker(ax, cal_r_acr, 'blue', 'r_acr')
# _plot_marker(ax, cal_ss, 'blue', 'ss')
# _plot_marker(ax, cal_c7, 'blue', 'c7')
# _plot_marker(ax, cal_xp, 'blue', 'xp')
# _plot_marker(ax, cal_ua3, 'blue', 'ua3')
# _plot_marker(ax, cal_ua2, 'blue', 'ua2')
# _plot_marker(ax, cal_ua1, 'blue', 'ua1')
# _plot_marker(ax, cal_fa3, 'blue', 'fa3')
# _plot_marker(ax, cal_fa2, 'blue', 'fa2')
# _plot_marker(ax, cal_fa1, 'blue', 'fa1')
# _plot_marker(ax, cal_rs, 'blue', 'rs')
# _plot_marker(ax, cal_us, 'blue', 'us')
# _plot_marker(ax, cal_me, 'blue', 'me')
# _plot_marker(ax, cal_le, 'blue', 'le')
# _plot_marker(ax, cal_mcp2, 'blue', 'mcp2')
# _plot_marker(ax, cal_mcp5, 'blue', 'mcp5')
# _plot_coordinate_system(ax, cal_fa1, (np.stack((cal_fa_x, cal_fa_y, cal_fa_z), axis = 1)))
# _plot_coordinate_system(ax, cal_fa1, (np.stack((cal_ua_x, cal_ua_y, cal_ua_z), axis = 1)))
# _plot_coordinate_system(ax, cal_fa1, (np.stack((cal_chest_x, cal_chest_y, cal_chest_z), axis = 1)))
# ax.quiver(
#     cal_us[0],
#     cal_us[1],
#     cal_us[2],
#     cal_me[0] - cal_us[0],
#     cal_me[1] - cal_us[1],
#     cal_me[2] - cal_us[2],
#     arrow_length_ratio=0.1,
#     color="purple",
# )
#     # Plot vector from elbow midpoint to r_acr
# ax.quiver(
#     cal_me[0],
#     cal_me[1],
#     cal_me[2],
#     cal_r_acr[0] - cal_me[0],
#     cal_r_acr[1] - cal_me[1],
#     cal_r_acr[2] - cal_me[2],
#     arrow_length_ratio=0.1,
#     color="purple",
# )
# ax.quiver(
#     cal_r_acr[0],
#     cal_r_acr[1],
#     cal_r_acr[2],
#     cal_ss[0] - cal_r_acr[0],
#     cal_ss[1] - cal_r_acr[1],
#     cal_ss[2] - cal_r_acr[2],
#     arrow_length_ratio=0.1,
#     color="purple",
# )
# # plt.show()

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


# ''' TASK TRIAL PROCESSING'''
# reads csv into a data frame
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV" 
trial_file = f"{trial_folder}/d_{sub_num}_EASY_LOW_5.tsv"
trial_raw = pd.read_csv(trial_file, sep='\t', header = 11) #sets csv to df



trial_markers = {
    "mcp2": trial_raw[["MCP2 X", "MCP2 Y", "MCP2 Z"]].values,
    "mcp5": trial_raw[["MCP5 X", "MCP5 Y", "MCP5 Z"]].values,
    "rs": trial_raw[["RS X", "RS Y", "RS Z"]].values,
    "us": trial_raw[["US X", "US Y", "US Z"]].values,
    "le": trial_raw[["LE X", "LE Y", "LE Z"]].values,
    "me": trial_raw[["ME X", "ME Y", "ME Z"]].values,
    "r_acr": trial_raw[["R_ACR X", "R_ACR Y", "R_ACR Z"]].values,
    "ss": trial_raw[["SS X", "SS Y", "SS Z"]].values,
    "xp": trial_raw[["XP X", "XP Y", "XP Z"]].values,
    "c7": trial_raw[["C7 X", "C7 Y", "C7 Z"]].values,
    "l_acr": trial_raw[["L_ACR X", "L_ACR Y", "L_ACR Z"]].values,
}

    #dictionary containing trialibration individual markers
trial_clusters = {
    "fa1": trial_raw[["FA1 X", "FA1 Y", "FA1 Z"]].values,
    "fa2": trial_raw[["FA2 X", "FA2 Y", "FA2 Z"]].values,
    "fa3": trial_raw[["FA3 X", "FA3 Y", "FA3 Z"]].values,
    "ua1": trial_raw[["UA1 X", "UA1 Y", "UA1 Z"]].values,
    "ua2": trial_raw[["UA2 X", "UA2 Y", "UA2 Z"]].values,
    "ua3": trial_raw[["UA3 X", "UA3 Y", "UA3 Z"]].values,
    "chest1": trial_raw[["CHEST1 X", "CHEST1 Y", "CHEST1 Z"]].values,
    "chest2": trial_raw[["CHEST2 X", "CHEST2 Y", "CHEST2 Z"]].values,
    "chest3": trial_raw[["CHEST3 X", "CHEST3 Y", "CHEST3 Z"]].values,
    "chest4": trial_raw[["CHEST4 X", "CHEST4 Y", "CHEST4 Z"]].values,
    "chest5": trial_raw[["CHEST5 X", "CHEST5 Y", "CHEST5 Z"]].values,
}
    # creating arrays containing marker indexing
# trial cluster marker arrays
trial_fa1 = trial_clusters['fa1']
trial_fa2 = trial_clusters['fa2']
trial_fa3 = trial_clusters['fa3']
trial_ua1 = trial_clusters['ua1']
trial_ua2 = trial_clusters['ua2']
trial_ua3 = trial_clusters['ua3']
trial_chest1 = trial_clusters['chest1']
trial_chest2 = trial_clusters['chest2']
trial_chest3 = trial_clusters['chest3']
trial_chest4 = trial_clusters['chest4']
trial_chest5 = trial_clusters['chest5']

trial_mcp2 = trial_markers['mcp2']
trial_mcp5 = trial_markers['mcp5']
trial_rs = trial_markers['rs']
trial_us = trial_markers['us']
trial_le =  trial_markers['le']
trial_me =  trial_markers['me']
trial_r_acr = trial_markers['r_acr']
trial_ss =  trial_markers['ss']
trial_xp =  trial_markers['xp']
trial_c7 =  trial_markers['c7']
trial_l_acr = trial_markers['l_acr']


# DEFINE LCS AND UNIT VECTORS FOR TASK CLUSTERS

# use imported task trial to iterate through cluster markers
# remaking coordinate systems using task clusters
trial_frame_count = len(trial_raw) # use this throughout to create array shapes and for loops

# create empty lists to store LCS vectors
# hand vectors
hand_trial_x_cluster_lcs = []
hand_trial_y_cluster_lcs = []
hand_trial_z_cluster_lcs = []

# forearm vectors 
fa_trial_x_cluster_lcs = []
fa_trial_y_cluster_lcs = []
fa_trial_z_cluster_lcs = []

# upper arm vectors
ua_trial_x_cluster_lcs = []
ua_trial_y_cluster_lcs = []
ua_trial_z_cluster_lcs = []

# chest vectors
chest_trial_x_cluster_lcs = []
chest_trial_y_cluster_lcs = []
chest_trial_z_cluster_lcs = []



#iterate through the length of the trial to create a LCS for each frame
#creating LCS using the clusters | hand uses indiviudal markers
''' size = (length of trial,3)''' # CURRENTLY (3,)
chest_trial_lcs = np.zeros((trial_frame_count, 3, 3))
ua_trial_lcs = np.zeros((trial_frame_count, 3, 3))
fa_trial_lcs = np.zeros((trial_frame_count, 3, 3))
hand_trial_lcs = np.zeros((trial_frame_count, 3, 3))
for frame in range(trial_frame_count):
     #Chest, O = chest 5, +Z towards right, +Y towards head, +X forward  
    chest_trial_z_cluster_lcs = ((trial_chest4[frame, :] - trial_chest5[frame, :])) /(np.linalg.norm(trial_chest4[frame, :] - trial_chest5[frame, :]))
    chest_trial_temp_cluster_lcs = (trial_chest3[frame, :] - trial_chest5[frame, :]) / (np.linalg.norm(trial_chest3[frame, :] - trial_chest5[frame, :]))
    chest_trial_x_cluster_lcs = np.cross(chest_trial_temp_cluster_lcs, chest_trial_z_cluster_lcs) / np.linalg.norm(np.cross(chest_trial_temp_cluster_lcs, chest_trial_z_cluster_lcs))
    chest_trial_y_cluster_lcs = np.cross(chest_trial_z_cluster_lcs, chest_trial_x_cluster_lcs) / np.linalg.norm(np.cross(chest_trial_z_cluster_lcs, chest_trial_x_cluster_lcs))
    chest_trial_lcs[frame, :, :] = np.stack((chest_trial_x_cluster_lcs, chest_trial_y_cluster_lcs, chest_trial_z_cluster_lcs), axis=0)
    # Right Upper Arm, tip facing POSTERIOR, O = ua1, +Y = superior, +X = anterior, +Z = lateral
    ua_trial_y_cluster_lcs = ((trial_ua3[frame, :] - trial_ua1[frame, :])) /(np.linalg.norm(trial_ua3[frame, :] - trial_ua1[frame, :]))
    ua_trial_temp_cluster_lcs = (trial_ua2[frame, :] - trial_ua1[frame, :]) / (np.linalg.norm(trial_ua2[frame, :] - trial_ua1[frame, :]))
    ua_trial_z_cluster_lcs = np.cross(ua_trial_y_cluster_lcs, ua_trial_temp_cluster_lcs) / np.linalg.norm(np.cross(ua_trial_y_cluster_lcs, ua_trial_temp_cluster_lcs))
    ua_trial_x_cluster_lcs = np.cross(ua_trial_y_cluster_lcs, ua_trial_z_cluster_lcs) / np.linalg.norm(np.cross(ua_trial_y_cluster_lcs, ua_trial_z_cluster_lcs))
    ua_trial_lcs[frame, :, :] = np.stack((ua_trial_x_cluster_lcs, ua_trial_y_cluster_lcs, ua_trial_z_cluster_lcs), axis=0)
    #Forearm, O = fa1, y = towards elbow, z = posterior, x = lateral
    fa_trial_y_cluster_lcs = ((trial_fa3[frame, :] - trial_fa1[frame, :])) /(np.linalg.norm(trial_fa3[frame, :] - trial_fa1[frame, :]))
    fa_trial_temp_cluster_lcs = (trial_fa2[frame,:] - trial_fa1[frame, :]) / (np.linalg.norm(trial_fa2[frame, :] - trial_fa1[frame, :]))
    fa_trial_x_cluster_lcs = np.cross(fa_trial_y_cluster_lcs, fa_trial_temp_cluster_lcs) / np.linalg.norm(np.cross(fa_trial_y_cluster_lcs, fa_trial_temp_cluster_lcs))
    fa_trial_z_cluster_lcs = np.cross(fa_trial_x_cluster_lcs, fa_trial_y_cluster_lcs) / np.linalg.norm(np.cross(fa_trial_x_cluster_lcs, fa_trial_y_cluster_lcs))
    fa_trial_lcs[frame, :, :] = np.stack((fa_trial_x_cluster_lcs, fa_trial_y_cluster_lcs, fa_trial_z_cluster_lcs), axis=0)
    #Hand, O = mcp2, Y = towards wrist, Z = radial X = dorsal
    hand_trial_y_cluster_lcs = ((trial_rs[frame, :] - trial_mcp2[frame, :])) /(np.linalg.norm(trial_rs[frame, :] - trial_mcp2[frame, :]))
    hand_trial_temp_cluster_lcs = (trial_us[frame,: ] - trial_mcp2[frame, :]) / (np.linalg.norm(trial_us[frame, :] - trial_mcp2[frame, :]))
    hand_trial_x_cluster_lcs = np.cross(hand_trial_y_cluster_lcs, hand_trial_temp_cluster_lcs) / np.linalg.norm(np.cross(hand_trial_y_cluster_lcs, hand_trial_temp_cluster_lcs))
    hand_trial_z_cluster_lcs = np.cross(hand_trial_x_cluster_lcs, hand_trial_y_cluster_lcs) / np.linalg.norm(np.cross(hand_trial_x_cluster_lcs, hand_trial_y_cluster_lcs))
    hand_trial_lcs[frame, :, :] = np.stack((hand_trial_x_cluster_lcs, hand_trial_y_cluster_lcs, hand_trial_z_cluster_lcs), axis=0)

# DEFINE ROTATION MATRIX (GLOBAL TO LOCAL) FOR TASK CLUSTER/SEGMENTS

# hand_trial_GRL = compute_GRL_rotation_matrix(hand_trial_x_frame, hand_trial_y_frame, hand_trial_z_frame)
# fa_trial_GRL = compute_GRL_rotation_matrix(fa_trial_x_frame, fa_trial_y_frame, fa_trial_z_frame)
# ua_trial_GRL = compute_GRL_rotation_matrix(ua_trial_x_frame, ua_trial_y_frame, ua_trial_z_frame)
# chest_trial_GRL = compute_GRL_rotation_matrix(chest_trial_x_frame, chest_trial_y_frame, chest_trial_z_frame)
# le_ua_trial_virtual = []
# me_ua_trial_virtual = []
# racr_ua_trial_virtual = []
# # RECREATE MARKERS
#     # not recreating mcp2, mcp5, rs or us 
#     # create virtual markers and rotate them back to global
#         # np.dot(LCS inv, point) + LCSorigin
# for frame in range(trial_frame_count):
    #recreate upper arm from UA1 - ME, LE, R_ACR
le_ua_trial_virtual = (np.dot(np.linalg.inv(ua_trial_lcs), le_ua) + trial_ua1)
me_ua_trial_virtual = (np.dot(np.linalg.inv(ua_trial_lcs), me_ua) + trial_ua1)
racr_ua_trial_virtual = (np.dot(np.linalg.inv(ua_trial_lcs), r_acr_ua) + trial_ua1)

#recreate torso from CHEST5 - SS XP 
ss_chest_trial_virtual = (np.dot(np.linalg.inv(chest_trial_lcs), ss_chest5) + trial_chest5)
xp_chest_trial_virtual = (np.dot(np.linalg.inv(chest_trial_lcs), xp_chest5) + trial_chest5)
c7_chest_trial_virtual = (np.dot(np.linalg.inv(chest_trial_lcs), c7_chest5) + trial_chest5)


# # PLOT TRIAL MARKERS, VIRTUAL MARKERS AND CLUSTER LCS
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(0,2000)
ax.set_ylim(0,2000)
ax.set_zlim(0,2000)
_plot_marker(ax, trial_chest5[500,:], 'blue', 'chest5')
_plot_marker(ax, trial_chest4[500,:], 'blue', 'chest4')
_plot_marker(ax, trial_chest3[500,:], 'blue', 'chest3')
_plot_marker(ax, trial_chest2[500,:], 'blue', 'chest2')
_plot_marker(ax, trial_chest1[500,:], 'blue', 'chest1')
_plot_marker(ax, trial_r_acr[500,:], 'blue', 'r_acr')
_plot_marker(ax, trial_ss[500,:], 'blue', 'ss')
_plot_marker(ax, trial_c7[500,:], 'blue', 'c7')
_plot_marker(ax, trial_xp[500,:], 'blue', 'xp')
_plot_marker(ax, trial_ua3[500,:], 'blue', 'ua3')
_plot_marker(ax, trial_ua2[500,:], 'blue', 'ua2')
_plot_marker(ax, trial_ua1[500,:], 'blue', 'ua1')
_plot_marker(ax, trial_fa3[500,:], 'blue', 'fa3')
_plot_marker(ax, trial_fa2[500,:], 'blue', 'fa2')
_plot_marker(ax, trial_fa1[500,:], 'blue', 'fa1')
_plot_marker(ax, trial_rs[500,:], 'blue', 'rs')
_plot_marker(ax, trial_us[500,:], 'blue', 'us')
_plot_marker(ax, trial_me[500,:], 'blue', 'me')
_plot_marker(ax, trial_le[500,:], 'blue', 'le')
_plot_marker(ax, trial_mcp2[500,:], 'blue', 'mcp2')
_plot_marker(ax, trial_mcp5[500,:], 'blue', 'mcp5')
_plot_marker(ax, le_ua_trial_virtual[500,:], 'red', 'le virt')
_plot_marker(ax, me_ua_trial_virtual[500,:], 'red', 'me virt')
_plot_marker(ax, racr_ua_trial_virtual[500,:], 'red', 'racr virt')
_plot_marker(ax, ss_chest_trial_virtual[500,:], 'red', 'ss virt')
_plot_marker(ax, c7_chest_trial_virtual[500,:], 'red', 'c7 virt')
_plot_marker(ax, xp_chest_trial_virtual[500,:], 'red', 'xp virt')
_plot_coordinate_system(ax, trial_chest5,chest_trial_lcs)
_plot_coordinate_system(ax, trial_fa1, fa_trial_lcs)
_plot_coordinate_system(ax, trial_ua1, ua_trial_lcs)
ax.quiver(
    trial_us[500, 0],
    trial_us[500, 1],
    trial_us[500, 2],
    trial_me[500, 0] - trial_us[500, 0],
    trial_me[500, 1] - trial_us[500, 1],
    trial_me[500, 2] - trial_us[500, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
    # Plot vector from elbow midpoint to r_acr
ax.quiver(
    trial_me[500, 0],
    trial_me[500, 1],
    trial_me[500, 2],
    trial_r_acr[500, 0] - trial_me[500, 0],
    trial_r_acr[500, 1] - trial_me[500, 1],
    trial_r_acr[500, 2] - trial_me[500, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
ax.quiver(
    trial_r_acr[500, 0],
    trial_r_acr[500, 1],
    trial_r_acr[500, 2],
    trial_ss[500, 0] - trial_r_acr[500, 0],
    trial_ss[500, 1] - trial_r_acr[500, 1],
    trial_ss[500, 2] - trial_r_acr[500, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
plt.show()



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
#   ss_trial_filtered[:,i] = butter_low(trial_ss[:,i]) # testing non-virtual
#   xp_trial_filtered[:,i] = butter_low(trial_xp[:,i])# testing non-virtual
#   c7_trial_filtered[:,i] = butter_low(trial_c7[:,i])# testing non-virtual
  rs_trial_filtered[:,i] = butter_low(trial_rs[:,i])
  us_trial_filtered[:,i] = butter_low(trial_us[:,i])
  mcp2_trial_filtered[:,i] = butter_low(trial_mcp2[:,i])
  mcp5_trial_filtered[:,i] = butter_low(trial_mcp5[:,i])
  
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
  ejc[:,i] = (le_trial_filtered[:,i] + me_trial_filtered[:,i])/2
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
#hand segment vectors
hand_seg_trial_x = np.empty_like(le_trial_filtered)
hand_seg_trial_y = np.empty_like(le_trial_filtered)
hand_seg_trial_z = np.empty_like(le_trial_filtered)
hand_seg_trial_temp = np.empty_like(le_trial_filtered)

hand_seg_trial_y_norm  = np.empty_like(le_trial_filtered)
hand_seg_trial_x_norm  = np.empty_like(le_trial_filtered)
hand_seg_trial_z_norm = np.empty_like(le_trial_filtered)

hand_seg_trial_lcs = np.zeros((trial_frame_count, 3, 3))
fa_seg_trial_lcs = np.zeros((trial_frame_count, 3, 3))
ua_seg_trial_lcs = np.zeros((trial_frame_count, 3, 3))
thrx_seg_trial_lcs = np.zeros((trial_frame_count, 3, 3))

thrx_y = np.tile([0, 1, 0], (trial_frame_count, 1))
for frame in range(trial_frame_count):
#forearm
    fa_seg_trial_y[frame,:] = ((ejc[frame,:] - us_trial_filtered[frame,:])) 
    fa_seg_trial_temp[frame,:] = (rs_trial_filtered[frame,:] - us_trial_filtered[frame,:]) 
    fa_seg_trial_x[frame,:] = np.cross(fa_seg_trial_y[frame,:], fa_seg_trial_temp[frame,:])
    fa_seg_trial_z[frame,:] = np.cross(fa_seg_trial_x[frame,:], fa_seg_trial_y[frame,:]) 

    fa_seg_trial_y_norm[frame,:] = fa_seg_trial_y[frame,:]/(np.linalg.norm(fa_seg_trial_y[frame,:]))
    fa_seg_trial_x_norm[frame,:] = fa_seg_trial_x[frame,:]/(np.linalg.norm(fa_seg_trial_x[frame,:]))
    fa_seg_trial_z_norm[frame,:] = fa_seg_trial_z[frame,:]/(np.linalg.norm(fa_seg_trial_z[frame,:]))
    # fa_seg_trial_lcs[frame,:, :] = np.stack((fa_seg_trial_x_norm, fa_seg_trial_y_norm, fa_seg_trial_z_norm), axis=1)
    #upper arm
    ua_seg_trial_y[frame,:] = ((sjc[frame,:] - ejc[frame,:])) 
    ua_seg_trial_temp[frame,:] = (le_trial_filtered[frame,:] - ejc[frame,:]) 
    ua_seg_trial_x[frame,:] = np.cross(ua_seg_trial_y[frame,:], ua_seg_trial_temp[frame,:]) 
    ua_seg_trial_z[frame,:] = np.cross(ua_seg_trial_x[frame,:], ua_seg_trial_y[frame,:]) 

    ua_seg_trial_y_norm[frame,:] = ua_seg_trial_y[frame,:] / (np.linalg.norm(ua_seg_trial_y[frame,:]))
    ua_seg_trial_x_norm[frame,:] = ua_seg_trial_x[frame,:] / (np.linalg.norm(ua_seg_trial_x[frame,:]))
    ua_seg_trial_z_norm[frame,:] = ua_seg_trial_z[frame,:] / (np.linalg.norm(ua_seg_trial_z[frame,:]))
    # ua_seg_trial_lcs[frame,:, :] = np.stack((ua_seg_trial_x_norm, ua_seg_trial_y_norm, ua_seg_trial_z_norm), axis=0)
    #thorax, origin set to ss
    # thrx_y = np.tile([0, 1, 0], (trial_frame_count, 1)) #np.tile creates an array with the repeating number, rows = frames
    thrx_seg_trial_y[frame,:] = thrx_y[frame,:]
    thrx_seg_trial_temp[frame,:] = (c7_trial_filtered[frame, :] - ss_trial_filtered[frame, :])
    thrx_seg_trial_z[frame,:] = np.cross(thrx_seg_trial_y[frame, :], thrx_seg_trial_temp[frame, :]) 
    thrx_seg_trial_x[frame,:] = np.cross(thrx_seg_trial_y[frame, :], thrx_seg_trial_z[frame, :])
 
    thrx_seg_trial_y_norm[frame,:] = thrx_seg_trial_y[frame,:] / (np.linalg.norm(thrx_seg_trial_y[frame,:]))
    thrx_seg_trial_x_norm[frame,:] = thrx_seg_trial_x[frame,:] / (np.linalg.norm(thrx_seg_trial_x[frame,:]))
    thrx_seg_trial_z_norm[frame,:] = thrx_seg_trial_z[frame,:] / (np.linalg.norm(thrx_seg_trial_z[frame,:]))
    # thrx_seg_trial_lcs[frame, :, :] = np.stack((thrx_seg_trial_x_norm, thrx_seg_trial_y_norm, thrx_seg_trial_z_norm), axis=0)

    hand_seg_trial_y[frame,:] = (hand_origin_prox[frame,:] - hand_origin[frame,:])
    hand_seg_trial_temp[frame,:] = (mcp2_trial_filtered[frame,:] - hand_origin[frame,:])
    hand_seg_trial_x[frame,:] = np.cross(hand_seg_trial_y[frame,:], hand_seg_trial_temp[frame,:]) 
    hand_seg_trial_z[frame,:] = np.cross(hand_seg_trial_x[frame,:], hand_seg_trial_y[frame,:]) 

    hand_seg_trial_y_norm[frame,:] = hand_seg_trial_y[frame,:] / (np.linalg.norm(hand_seg_trial_y[frame,:]))
    hand_seg_trial_x_norm[frame,:] = hand_seg_trial_x[frame,:] / (np.linalg.norm(hand_seg_trial_x[frame,:]))
    hand_seg_trial_z_norm[frame,:] = hand_seg_trial_z[frame,:] / (np.linalg.norm(hand_seg_trial_z[frame,:]))
    # hand_seg_trial_lcs[frame, :, :] = np.stack((hand_seg_trial_x_norm, hand_seg_trial_y_norm, hand_seg_trial_z_norm), axis=0)


# Visual checks for anatomical lcs

# PLOT TRIAL MARKERS, VIRTUAL MARKERS AND CLUSTER LCS
    # virtual-filtered markers 
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(0,2000)
ax.set_ylim(0,2000)
ax.set_zlim(0,2000)
_plot_marker(ax, mcp2_trial_filtered[50,:], 'blue', 'mcp2')
_plot_marker(ax, mcp5_trial_filtered[50,:], 'blue', 'mcp5')
_plot_marker(ax, racr_trial_filtered[50,:], 'blue', 'r_acr')
_plot_marker(ax, rs_trial_filtered[50,:], 'blue', 'rs')
_plot_marker(ax, us_trial_filtered[50,:], 'blue', 'us')
_plot_marker(ax, me_trial_filtered[50,:], 'blue', 'me')
_plot_marker(ax, le_trial_filtered[50,:], 'blue', 'le')
_plot_marker(ax, c7_trial_filtered[50,:], 'blue', 'c7')
_plot_marker(ax, ss_trial_filtered[50,:], 'blue', 'ss')
_plot_coordinate_system(ax, trial_ss[50,:], (thrx_seg_trial_x_norm[frame],thrx_seg_trial_y_norm[frame],thrx_seg_trial_z_norm[frame]))
# _plot_coordinate_system(ax, trial_chest5[50,:], chest_trial_lcs)
_plot_coordinate_system(ax, trial_fa1[50,:], (fa_seg_trial_x_norm[frame],fa_seg_trial_y_norm[frame],fa_seg_trial_z_norm[frame]))
_plot_coordinate_system(ax, trial_ua1[50,:], (ua_seg_trial_x_norm[frame],ua_seg_trial_y_norm[frame],ua_seg_trial_z_norm[frame]))
_plot_coordinate_system(ax, hand_origin[50,:], (hand_seg_trial_x_norm[frame],hand_seg_trial_y_norm[frame],hand_seg_trial_z_norm[frame]))
ax.quiver(
    wjc[50, 0],
    wjc[50, 1],
    wjc[50, 2],
    ejc[50, 0] - wjc[50, 0],
    ejc[50, 1] - wjc[50, 1],
    ejc[50, 2] - wjc[50, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
    # Plot vector from elbow midpoint to r_acr
ax.quiver(
    ejc[50, 0],
    ejc[50, 1],
    ejc[50, 2],
    sjc[50, 0] - ejc[50, 0],
    sjc[50, 1] - ejc[50, 1],
    sjc[50, 2] - ejc[50, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
ax.quiver(
    sjc[50, 0],
    sjc[50, 1],
    sjc[50, 2],
    ss_trial_filtered[50, 0] - sjc[50, 0],
    ss_trial_filtered[50, 1] - sjc[50, 1],
    ss_trial_filtered[50, 2] - sjc[50, 2],
    arrow_length_ratio=0.1,
    color="purple",
)
plt.show()


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
    # Set forearm pre-dcm to the unit vectors from the anatomical lcs
    fa_dcm_ijk = np.vstack((
        fa_seg_trial_x_norm[frame],
        fa_seg_trial_y_norm[frame],
        fa_seg_trial_z_norm[frame])
    )
    fa_dcm_ijk_t = np.transpose(fa_dcm_ijk)
        # Set Uppe Arm pre-dcm to the unit vectors from the anatomical lcs
    hand_dcm_ijk = np.vstack((
        hand_seg_trial_x_norm[frame],
        hand_seg_trial_y_norm[frame],
        hand_seg_trial_z_norm[frame])
    )
    hand_dcm_ijk_t = np.transpose(hand_dcm_ijk)
    # dot product between the two anatomical LCS 
    wrist_dcm = np.array( 
        [
        [ 
                np.dot(hand_dcm_ijk_t[0], fa_dcm_ijk[0]),
                np.dot(hand_dcm_ijk_t[0], fa_dcm_ijk[1]),
                np.dot(hand_dcm_ijk_t[0], fa_dcm_ijk[2])
            ],
            [
                np.dot(hand_dcm_ijk_t[1], fa_dcm_ijk[0]),
                np.dot(hand_dcm_ijk_t[1], fa_dcm_ijk[1]),
                np.dot(hand_dcm_ijk_t[1], fa_dcm_ijk[2]),
            ],
            [
                np.dot(hand_dcm_ijk_t[2], fa_dcm_ijk[0]),
                np.dot(hand_dcm_ijk_t[2], fa_dcm_ijk[1]),
                np.dot(hand_dcm_ijk_t[2], fa_dcm_ijk[2]),
            ] 
        ]
    )
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

# plt.plot(x, alphadeg_wrist, label = "alpha (p+/s-)", linestyle="-")
# plt.plot(x, betadeg_wrist, label = "beta (rd-/ud+)", linestyle="--")
# plt.plot(x, gammadeg_wrist, label = "gamma (f+/e-)", linestyle=":")
# plt.legend()
# plt.title('wrist angles')
# plt.show()

# ''' ELBOW '''
#ulna relative to humerus (elbow) Z-X-Y
    # alphaHF (Z) = flexion (+)/ hyperextension (-)  
    # gammaHF (Y) = axial rotation of the forearm | pronation (+) / supination (-) 
    # betaHF (X) = carrying angle, passive response to flex/ext, rarely reported
# np.set_printoptions(precision = 5)
DCM_elbow = np.zeros((trial_frame_count,3,3))
alpha_elbow = np.zeros((trial_frame_count,1))
beta_elbow = np.zeros((trial_frame_count,1))
gamma_elbow = np.zeros((trial_frame_count,1))

for frame in range(trial_frame_count):
        # Set forearm pre-dcm to the unit vectors from the anatomical lcs
    fa_dcm_ijk = np.vstack((fa_seg_trial_x_norm[frame], fa_seg_trial_y_norm[frame], fa_seg_trial_z_norm[frame]))
    fa_dcm_ijk_t = np.transpose(fa_dcm_ijk)
        # Set Uppe Arm pre-dcm to the unit vectors from the anatomical lcs
    ua_dcm_ijk = np.vstack((
        ua_seg_trial_x_norm[frame],
        ua_seg_trial_y_norm[frame],
        ua_seg_trial_z_norm[frame])
    )
    ua_dcm_ijk_t = np.transpose(ua_dcm_ijk)
    # dot product between the two anatomical LCS 
    elbow_dcm = np.array( 
        [
        [ 
                np.dot(fa_dcm_ijk_t[0], ua_dcm_ijk[0]),
                np.dot(fa_dcm_ijk_t[0], ua_dcm_ijk[1]),
                np.dot(fa_dcm_ijk_t[0], ua_dcm_ijk[2])
            ],
            [
                np.dot(fa_dcm_ijk_t[1], ua_dcm_ijk[0]),
                np.dot(fa_dcm_ijk_t[1], ua_dcm_ijk[1]),
                np.dot(fa_dcm_ijk_t[1], ua_dcm_ijk[2]),
            ],
            [
                np.dot(fa_dcm_ijk_t[2], ua_dcm_ijk[0]),
                np.dot(fa_dcm_ijk_t[2], ua_dcm_ijk[1]),
                np.dot(fa_dcm_ijk_t[2], ua_dcm_ijk[2]),
            ] 
        ]
    )
    
    #   #row then col
    beta_elbow[frame,:] = np.arcsin(elbow_dcm[1,2])
    alpha_elbow[frame, :] = np.arccos((elbow_dcm[1,1])/np.cos(beta_elbow[frame]))
    gamma_elbow[frame,:] = np.arccos((elbow_dcm[2,2])/np.cos(beta_elbow[frame]))

    # print(beta_elbow)
    alphadeg_elbow = np.degrees(alpha_elbow)
    betadeg_elbow = np.degrees(beta_elbow)
    gammadeg_elbow = np.degrees(gamma_elbow)

# plt.plot(x, alphadeg_elbow, label = "alpha (+f/e)", linestyle="-")
# plt.plot(x, betadeg_elbow, label = "beta (carrying angle)", linestyle="--")
# plt.plot(x, gammadeg_elbow, label = "gamma (p+/s-)", linestyle=":")
# plt.legend()
# plt.title('elbow angles')
# plt.show()

''' SHOULDER '''
# humerus relative to the thorax
    # gamma1HT (Y1) = plane of elevation | 0 abduction, 90 forward flexion
    # betaHT (X) = elevation (negative)
    # gamma2HT (Y2) = axial rotation | internal + / external - 

DCM_shoulder = np.zeros((trial_frame_count,3,3))
gamma1_shoulder = np.zeros((trial_frame_count,1))
beta_shoulder = np.zeros((trial_frame_count,1))
gamma2_shoulder = np.zeros((trial_frame_count,1))
gamma_shoulder = np.zeros((trial_frame_count,1))
alpha_shoulder =np.zeros((trial_frame_count,1))

for frame in range(trial_frame_count):
        # Set Upper Arm pre-dcm to the unit vectors from the anatomical lcs
    ua_dcm_ijk = np.vstack((
        ua_seg_trial_x_norm[frame],
        ua_seg_trial_y_norm[frame],
        ua_seg_trial_z_norm[frame])
    )
    ua_dcm_ijk_t = np.transpose(ua_dcm_ijk)

    thrx_dcm_ijk =  np.vstack((
        thrx_seg_trial_x_norm[frame],
        thrx_seg_trial_y_norm[frame],
        thrx_seg_trial_z_norm[frame])
    )
    # dot product between the two anatomical LCS 
    shoulder_dcm = np.array( 
        [
        [ 
                np.dot(ua_dcm_ijk_t[0], thrx_dcm_ijk[0]),
                np.dot(ua_dcm_ijk_t[0], thrx_dcm_ijk[1]),
                np.dot(ua_dcm_ijk_t[0], thrx_dcm_ijk[2])
            ],
            [
                np.dot(ua_dcm_ijk_t[1], thrx_dcm_ijk[0]),
                np.dot(ua_dcm_ijk_t[1], thrx_dcm_ijk[1]),
                np.dot(ua_dcm_ijk_t[1], thrx_dcm_ijk[2]),
            ],
            [
                np.dot(ua_dcm_ijk_t[2], thrx_dcm_ijk[0]),
                np.dot(ua_dcm_ijk_t[2], thrx_dcm_ijk[1]),
                np.dot(ua_dcm_ijk_t[2], thrx_dcm_ijk[2]),
            ] 
        ]
    )

    beta_shoulder[frame,:] = np.arccos((shoulder_dcm[1,1]))
    gamma1_shoulder[frame, :] = np.arcsin((shoulder_dcm[0,1])/np.sin(beta_shoulder[frame]))
    gamma2_shoulder[frame,:] = np.arcsin((shoulder_dcm[1,0])/np.sin(beta_shoulder[frame]))

    gamma1deg_shoulder = np.degrees(gamma1_shoulder)
    betadeg_shoulder = np.degrees(beta_shoulder)
    gamma2deg_shoulder = np.degrees(gamma2_shoulder)
    #Trying ZXY 
        # Calculate Beta (β), the rotation about the Z-axis
    beta_shoulder[frame, :] = np.arcsin(-shoulder_dcm[0, 2])

    # Calculate Alpha (α), the rotation about the X-axis
    alpha_shoulder[frame, :] = np.arctan2(shoulder_dcm[1, 2], shoulder_dcm[2, 2])

    # Calculate Gamma (γ), the rotation about the Y-axis
    gamma_shoulder[frame, :] = np.arctan2(shoulder_dcm[0, 1], shoulder_dcm[0, 0])

    # Convert radians to degrees for better interpretation
    alphadeg_shoulder = np.degrees(alpha_shoulder)
    betadeg_shoulder = np.degrees(beta_shoulder)
    gammadeg_shoulder = np.degrees(gamma_shoulder)

plt.plot(x, alphadeg_shoulder, label="alpha (rotation about X-axis)", linestyle="-")
plt.plot(x, betadeg_shoulder, label="beta (rotation about Z-axis)", linestyle="--")
plt.plot(x, gammadeg_shoulder, label="gamma (rotation about Y-axis)", linestyle=":")
plt.legend()
plt.title('Shoulder Angles (X-Z-Y Sequence)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.show()

plt.plot(x, gamma1_shoulder, label = "gamma1 (abd/ff)", linestyle="-")
plt.plot(x, betadeg_shoulder, label = "beta (elevation)", linestyle="--")
plt.plot(x, gamma2deg_shoulder, label = "gamma2 (ir+/er-)", linestyle=":")
plt.legend()
plt.title('Shoulder Angles (Y-X-Y)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.show()