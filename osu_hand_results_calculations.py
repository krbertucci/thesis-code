
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
import os

#'''IMPORT TRIAL FOLDER''' 
#update subject number
sub_num = "S08"
# set path for files
# trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"
trial_folder = f"D:/MSc_Thesis/OSU/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"
# sample rate
fs = 100

hand_outputs_df = pd.DataFrame(columns=['Subject' ,'Difficulty' , 'Sensitivity',
                                        'X Velocity Mean', 'X Velocity Max', 'X Velocity Min',
                                        'Z Velocity Mean', 'Z Velocoty Max', 'Z Velocity Min',
                                        'X Acceleration Mean', 'X Acceleration Max', 'X Acceleration Min',
                                        'Z Acceleration Mean', 'Z Acceleration Max', 'Z Acceleration Min'
                                           ])


difficulties = ['EASY']
sensitivities = ['PREF', 'LOW', 'HIGH']

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

velocities_outputs_list = [] 
accelerations_outputs_list = []
results =[]        
np.set_printoptions(suppress=True, precision=None, )
for difficulty in difficulties:
    for sensitivity in sensitivities:
        hand_x_vel_mean_arr = []
        hand_x_vel_max_arr = []
        hand_x_vel_min_arr = []
        hand_z_vel_mean_arr = []
        hand_z_vel_max_arr = []
        hand_z_vel_min_arr = []
        hand_x_acc_mean_arr = []
        hand_x_acc_max_arr = []
        hand_x_acc_min_arr = []
        hand_z_acc_mean_arr = []
        hand_z_acc_max_arr = []
        hand_z_acc_min_arr = []

        folder_prefix = f'd_{sub_num}_{difficulty}_{sensitivity}*.tsv'
        condition_trial_paths = glob.glob(f'{trial_folder}/{folder_prefix}')
        for condition_trial in condition_trial_paths:
            condition_trial_basename = os.path.basename(condition_trial).strip('.tsv')
            trial_raw = pd.read_csv(condition_trial, sep='\t', header = 11)
            if len(condition_trial_paths) == 0: continue
            
            trial_markers = {
                "mcp2": trial_raw[["MCP2 X", "MCP2 Y", "MCP2 Z"]].values,
                "mcp5": trial_raw[["MCP5 X", "MCP5 Y", "MCP5 Z"]].values,
                "rs": trial_raw[["RS X", "RS Y", "RS Z"]].values,
                "us": trial_raw[["US X", "US Y", "US Z"]].values,
            }

            trial_mcp2 = trial_markers['mcp2']
            trial_mcp5 = trial_markers['mcp5']
            trial_rs = trial_markers['rs']
            trial_us = trial_markers['us']
        
            trial_frame_count = len(trial_raw)

            rs_trial_filtered = np.empty_like(trial_rs) 
            us_trial_filtered = np.empty_like(trial_us)
            mcp2_trial_filtered = np.empty_like(trial_mcp2)
            mcp5_trial_filtered = np.empty_like(trial_mcp5)

            hand_origin = np.empty_like(trial_rs)
            hand_origin_dist = np.empty_like(trial_rs)
            hand_origin_prox = np.empty_like(trial_rs)   
          
            for i in range(trial_mcp2.shape[1]): # iterate through frame length at x y z
                rs_trial_filtered[:,i] = butter_low(trial_rs[:,i])
                us_trial_filtered[:,i] = butter_low(trial_us[:,i])
                mcp2_trial_filtered[:,i] = butter_low(trial_mcp2[:,i])
                mcp5_trial_filtered[:,i] = butter_low(trial_mcp5[:,i])
            
            for i in range(trial_mcp2.shape[1]): # iterate through frame length at x y z
                hand_origin_dist[:,i] = (mcp2_trial_filtered[:,i] + mcp5_trial_filtered[:,i])/2
                hand_origin_prox[:,i] = (rs_trial_filtered[:,i] + us_trial_filtered[:,i])/2
                hand_origin[:,i] = (hand_origin_dist[:,i] + hand_origin_prox[:,i])/2
            #break down hand origin into x and z components
            hand_origin_x = hand_origin[:,0]
            hand_origin_z = hand_origin[:,2]

            # print(len(hand_origin_x))
            #apply central differences technque
            t = 1/fs
            hand_x_vel = np.zeros(trial_frame_count)
            hand_z_vel = np.zeros(trial_frame_count)
            hand_x_vel_mps = np.zeros(trial_frame_count)
            hand_z_vel_mps = np.zeros(trial_frame_count)

            #displacement is in mm
            #time is 100 hz, so 0.01 seconds
            #vel is mm/0.01 seconds
            #

            # calculate velocity
            for frame in range(100, trial_frame_count - 100):
                hand_x_vel[frame] = (hand_origin_x[(frame + 1)] - hand_origin_x[(frame-1)])/(2*t)
                hand_z_vel[frame] = (hand_origin_z[(frame + 1)] - hand_origin_z[(frame-1)])/(2*t)
            
            hand_x_vel_mps = hand_x_vel / 10
            hand_z_vel_mps = hand_z_vel / 10

            # plt.plot(hand_x_vel_mps)
            # plt.title(f'hand x vel {condition_trial_basename}')
            # plt.xlabel('Frame')
            # plt.ylabel('velocity (m/s)')

            # plt.show()
            # calculate acceleration
            hand_x_acc = np.zeros(trial_frame_count)
            hand_z_acc = np.zeros(trial_frame_count)
            for frame in range(100, trial_frame_count - 100):
                hand_x_acc[frame] = (hand_x_vel_mps[(frame + 1)] - hand_x_vel_mps[(frame-1)])/(2*t)
                hand_z_acc[frame] = (hand_z_vel_mps[(frame + 1)] - hand_z_vel_mps[(frame-1)])/(2*t)
            #obtain output values
            hand_x_vel_mean_arr.append(np.mean(hand_x_vel_mps))
            hand_x_vel_max_arr.append(np.max(hand_x_vel_mps))
            hand_x_vel_min_arr.append(np.min(hand_x_vel_mps))
            hand_z_vel_mean_arr.append(np.mean(hand_z_vel_mps))
            hand_z_vel_max_arr.append(np.max(hand_z_vel_mps))
            hand_z_vel_min_arr.append(np.min(hand_z_vel_mps))
            #obtain output values
            hand_x_acc_mean_arr.append(np.mean(hand_x_acc))
            hand_x_acc_max_arr.append(np.max(hand_x_acc))
            hand_x_acc_min_arr.append(np.min(hand_x_acc))
            hand_z_acc_mean_arr.append(np.mean(hand_z_acc))
            hand_z_acc_max_arr.append(np.max(hand_z_acc))
            hand_z_acc_min_arr.append(np.min(hand_z_acc))
        #obtain absolute values of min to obtain peak
        hand_x_vel_min_arr_abs = np.absolute(hand_x_vel_min_arr)
        hand_z_vel_min_arr_abs = np.absolute(hand_z_vel_min_arr)
        hand_x_acc_min_arr_abs = np.absolute(hand_x_acc_min_arr)
        hand_z_acc_min_arr_abs = np.absolute(hand_z_acc_min_arr)

        hand_x_vel_mean = (np.mean(hand_x_vel_mean_arr))
        hand_x_vel_min_abs = np.max(hand_x_vel_min_arr_abs)
        hand_x_vel_max_temp = np.max(hand_x_vel_max_arr)
        hand_x_vel_max = np.max([hand_x_vel_max_temp, hand_x_vel_min_abs])
        hand_x_vel_min = (np.min(hand_x_vel_min_arr))

        hand_z_vel_mean = np.mean(hand_z_vel_mean_arr)
        hand_z_vel_min_abs = np.max(hand_z_vel_min_arr_abs)
        hand_z_vel_max_temp = np.max(hand_z_vel_max_arr)
        hand_z_vel_max = np.max([hand_z_vel_max_temp, hand_z_vel_min_abs])

        hand_x_acc_mean = np.mean(hand_x_acc_mean_arr)
        hand_x_acc_min_abs = np.max(hand_x_acc_min_arr_abs)
        hand_x_acc_max_temp = np.max(hand_x_acc_max_arr)
        hand_x_acc_max = np.max([hand_x_acc_max_temp, hand_x_acc_min_abs])

        hand_z_acc_mean = np.mean(hand_z_acc_mean_arr)
        hand_z_acc_min_abs = np.max(hand_z_acc_min_arr_abs)
        hand_z_acc_max_temp = np.max(hand_z_acc_max_arr)
        hand_z_acc_max = np.max([hand_z_acc_max_temp, hand_z_acc_min_abs])

        # hand_x_acc_mean = (np.mean(hand_x_acc_mean_arr))
        # hand_x_acc_max = (np.max(hand_x_acc_max_arr, np.absolute(hand_x_acc_min_arr)))
        # hand_x_acc_min = (np.min(hand_x_acc_min_arr))
        # hand_z_acc_mean = (np.mean(hand_z_acc_mean_arr))
        # hand_z_acc_max = (np.max(hand_z_acc_max_arr))
        # hand_z_acc_min = (np.min(hand_z_acc_min_arr))
        results.append({
            'Subject': sub_num,
            'Difficulty': difficulty,
            'Sensitivity': sensitivity,
            'X Velocity Mean': hand_x_vel_mean,
            'X Velocity Max': hand_x_vel_max,
            # 'X Velocity Min' : hand_x_vel_min,
            'Z Velocity Mean': hand_z_vel_mean,
            'Z Velocity Max': hand_z_vel_max,
            # 'Z Velocity Min': hand_z_vel_min,
            'X Acceleration Mean': hand_x_acc_mean,
            'X Acceleration Max': hand_x_acc_max,
            'Z Acceleration Mean': hand_z_acc_mean,
            'Z Acceleration Max': hand_z_acc_max,
        })


hand_outputs_df = pd.DataFrame(results)
            # plt.title(f'hand z vel {condition_trial_basename}')
            # plt.xlabel('Frame')
            # plt.ylabel('acceleration (m/s)')
            # # # plt.savefig(f"{figpath}/Shoulder Angles {condition_trial_basename} test.png")
            # # # plt.close()
            # plt.show()
    # print(velocities_outputs_list)
# #export to csv
hand_outputs_df.to_csv(f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/{sub_num}_hand_results.csv", float_format="%.8f")
hand_outputs_df.to_csv(f"D:/MSc_Thesis/OSU/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV/{sub_num}_hand_results.csv", float_format="%.8f")
hand_outputs_df.to_csv(f"D:/MSc_Thesis/OSU/Data_Analysis/Kinematics/OSU Hand Results/{sub_num}_hand_results.csv", float_format="%.8f")
print(f'{sub_num} data has been saved')
#
#only output / save the data after 1 second and before the last second
