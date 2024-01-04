#OSU TASK EMG - Kayla Russell-Bertucci (last modified: Dec 7, 2023)
# INPUTS = Task files
#   filters, removes bias, & full wave rectify for each muscle of task
#   normalize amplitude to peak MVC & time

# OUTPUT = 1. Processed task trials (individual & averaged)
#        = 2. %MVC peak for individual muscles in each mouse sensitivity and difficulty per participant

import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
import emg_signal_processing_functions
from emg_signal_processing_functions import process_muscle
# set subject number
sub_num = "S05"
subject_folder = r"C:\Users\kruss\OneDrive - University of Waterloo\Documents\OSU\Data\S05\Data_Raw\Trial_EMG\MVC" #sets path for files
subject_dirs = os.listdir(subject_folder) # creates a list of the files in the subject folder

#dir = glob.glob(f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data\S05\Data_Raw\Trial_EMG\Trial_EMG_Files")

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


# NAME CONVENTIONS
# EH - easy diff high sens 
# EL - easy diff low sens
# EP - easy diff pref sens
# HH - hard diff high sens
# HL - hard diff low sens
# HP - hard diff high sens
trialnum = 1 # number of trials 



for muscle, col in muscles.items(): 
    # set easy difficulty paths
    eh_trial_path = f"{subject_folder}\d_{sub_num}_EASY_HIGH_{trialnum}_a.tsv"
    # el_trial_path = f"{subject_folder}\d_{sub_num}_EASY_LOW_{trialnum}_a.tsv"
    # ep_trial_path = f"{subject_folder}\d_{sub_num}_EASY_PREF_{trialnum}_a.tsv"

    # # set hard difficulty paths
    # hh_trial_path = f"{subject_folder}\d_{sub_num}_HARD_HIGH_{trialnum}_a.tsv"
    # hl_trial_path = f"{subject_folder}\d_{sub_num}_HARD_LOW_{trialnum}_a.tsv"
    # hp_trial_path = f"{subject_folder}\d_{sub_num}_HARD_PREF_{trialnum}_a.tsv"
    
    eh_trial = pd.read_csv(eh_trial_path, sep='\t', header=13)
    # el_trial = pd.read_csv(el_trial_path, sep='\t', header=13)
    # ep_trial = pd.read_csv(ep_trial_path, sep='\t', header=13)

    # hh_trial = pd.read_csv(hh_trial_path, sep='\t', header=13)
    # hl_trial = pd.read_csv(hl_trial_path, sep='\t', header=13)
    # hp_trial = pd.read_csv(hp_trial_path, sep='\t', header=13)

    eh_signal = eh_trial[f"EMG {col}"].to_numpy()
    processed_muscle_signal_eh = process_muscle(eh_signal)

    print(processed_muscle_signal_eh)