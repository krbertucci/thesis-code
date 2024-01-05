#OSU TASK EMG - Kayla Russell-Bertucci (last modified: Jan 4, 2023)
# INPUTS
#   1. Subject Number
#   2. Task files
#   3. import MVC max from emg_mvc_processing excel output
# PURPOSE
#   filters, removes bias, & full wave rectify for each muscle of task
#   normalize amplitude to peak MVC
# OUTPUT = 1. Processed task trials (individual & averaged)
#        = 2. %MVC peak for individual muscles in each mouse sensitivity and difficulty per participant
#
# TRIAL NAMING CONVENTIONS
# eh - easy diff high sens 
# el - easy diff low sens
# ep - easy diff pref sens
# hh - hard diff high sens
# hl - hard diff low sens
# hp  - hard diff high sens


import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
# import emg_signal_processing_functions
# from emg_signal_processing_functions import process_muscle


def remove_bias(signal: np.ndarray):
    """Remove DC bias from signal.
    
    Args:
        signal: raw signal from MVC data.
    """
    MVC_signal_bias = np.mean(signal)
    MVC_signal_no_bias = signal - MVC_signal_bias
    return MVC_signal_no_bias

def butter_bandpass(signal: np.ndarray):
    """Function to design and apply Butterworth Filter

    Args:
        signal: This is the non-biased signal after remove_bias
    """
    # Butterworth bandpass filter specifications
    fs = 2000 # Sampling frequency
    a = 30 # Highpass cut off @ 30 Hz to remove HR (Drake and Callaghan 2006)
    b = 500 # Lowbass cutoff @ 500 Hz 
    Wn = np.array([a, b]) 
    sos = scipy.signal.butter(4, Wn, btype='bandpass', fs=fs, output='sos') 
    MVC_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    # MVC_filtered_signal = scipy.signal.filtfilt(b, a, signal)
    return MVC_filtered_signal

def full_wave_rectify(signal: np.ndarray):
    """Rectify filtered signal.

    Args:
        signal: Bandpass filtered signal.
    """
    MVC_FWR = np.absolute(signal)
    return MVC_FWR

def process_muscle(signal: np.ndarray):
    unbiased_signal = remove_bias(signal)
    bandpass_signal = butter_bandpass(unbiased_signal)
    rectified_signal = full_wave_rectify(bandpass_signal)

# set subject number
sub_num = "S05"
subject_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files" #sets path for files
#subject_dirs = os.listdir(subject_folder) # creates a list of the files in the subject folder

#set prefix and directory for each trial condition
    # easy diff pref sens
ep_prefix = f'd_{sub_num}_EASY_PREF*'
ep_dir = glob.glob(f'{subject_folder}/{ep_prefix}')
    # easy diff high sens
eh_prefix = f'd_{sub_num}_EASY_HIGH*'
eh_dir =  glob.glob(f'{subject_folder}/{eh_prefix}')
    # easy diff low sens
el_prefix = f'd_{sub_num}_EASY_LOW*'
el_dir = glob.glob(f'{subject_folder}/{el_prefix}')
    # hard diff pref sens
hp_prefix = f'd_{sub_num}_HARD_PREF*'
hp_dir = glob.glob(f'{subject_folder}/{hp_prefix}')
    # hard diff high sens
hh_prefix = f'd_{sub_num}_HARD_HIGH*'
hh_dir =  glob.glob(f'{subject_folder}/{hh_prefix}')
    # hard diff low sens
hl_prefix = f'd_{sub_num}_EASY_LOW*'
hl_dir = glob.glob(f'{subject_folder}/{hl_prefix}')

# Index trials in each directory to correspond with trial 1, 2 or 3
# ep trial index
ep_t1 = ep_dir[0] #easy pref trial 1
ep_t2 = ep_dir[1]
ep_t3 = ep_dir[2]
#el trial index
el_t1 = el_dir[0]
el_t2 = ep_dir[1]
el_t3 = ep_dir[2]
# print(ep_t1)
t1_check = pd.read_csv(el_t1, sep='\t', header=13)
# print(t1_check)

muscles = {
    "UTRAP": 1,
    "SUPRA": 2,
    "INFRA": 3,
    "TRICEP": 4,
    "BICEP": 5,
    "PEC(C)": 6,
    "PEC(S)": 7,
    "ADELT": 8,
    "MDELT": 9,
    "EDC": 10,
    "ECU": 11,
    "ECRB": 12
    } 

trial_muscle_max_arr = []
# one for loop for each condition
# ep loop
ep_muscle_max_arr = []
for muscle, col in muscles.items():
    ep_t1_csv = pd.read_csv(ep_t1, sep='\t', header=13)
    ep_t1_signal = ep_t1_csv[f"EMG {col}"].to_numpy()
    ep_t1_nobias = remove_bias(ep_t1_signal)
    ep_t1_butter = butter_bandpass(ep_t1_nobias)
    ep_t1_fwr = full_wave_rectify(ep_t1_butter)
    ep_t1_max = np.max(ep_t1_fwr)

    ep_t2_csv = pd.read_csv(ep_t2, sep='\t', header=13)
    ep_t2_signal = ep_t2_csv[f"EMG {col}"].to_numpy()
    ep_t2_nobias = remove_bias(ep_t2_signal)
    ep_t2_butter = butter_bandpass(ep_t2_nobias)
    ep_t2_fwr = full_wave_rectify(ep_t2_butter)
    ep_t2_max = np.max(ep_t2_fwr)

    ep_t3_csv = pd.read_csv(ep_t3, sep='\t', header=13)
    ep_t3_signal = ep_t3_csv[f"EMG {col}"].to_numpy()
    ep_t3_nobias = remove_bias(ep_t3_signal)
    ep_t3_butter = butter_bandpass(ep_t3_nobias)
    ep_t3_fwr = full_wave_rectify(ep_t3_butter)
    ep_t3_max = np.max(ep_t3_fwr)

    
    ep_muscle_max = max(ep_t1_max, ep_t2_max, ep_t3_max)
    ep_muscle_max_arr.append(ep_muscle_max)
    print(ep_muscle_max_arr)
    
#write into a csv
ep_muscle_max_arr = np.array(ep_muscle_max_arr)
ep_muscle_max_arr = np.expand_dims(ep_muscle_max_arr, axis=0)
subject_ep_trial_max_csv = pd.DataFrame(data=ep_muscle_max_arr, columns=muscles.keys())
print(subject_ep_trial_max_csv)

#     #plt.plot(ep_t1_fwr, label = muscle)
# plt.legend()
# plt.show()

# _, (ax1, ax2) = plt.subplot(3, sharex = True)
# ax1.plot(ep_t1_signal)
# ax2.plot(processed_ep_t1)
   
# print(processed_ep_t1)    
#     eh_signal = eh_trial[f"EMG {col}"].to_numpy()
#     processed_muscle_signal_eh = process_muscle(eh_signal)

#     print(processed_muscle_signal_eh)