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


import pandas as pd
import numpy as np
import glob
import emg_signal_processing_functions as emg_sp

# set subject number
sub_num = "S05"
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files" #sets path for files
# TODO: Step 1 Create new Folder similar to subject folder but for the MVC path.
mvc_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/MVC"

# dictionary with each condition and empty values - the values will be updated with the maxes
condition_maxs = {
    "EASY_PREF": [],
    "EASY_HIGH": [],
    "EASY_LOW": [],
    "HARD_PREF": [],
    "HARD_HIGH": [],
    "HARD_LOW": [],
}

normalized_maxs = {
    "EASY_PREF": [],
    "EASY_HIGH": [],
    "EASY_LOW": [],
    "HARD_PREF": [],
    "HARD_HIGH": [],
    "HARD_LOW": [],
}
# dictionary of the muscles and their columns
muscles = {
    "UTRAP": 1,
    "SUPRA": 2,
    "INFRA": 3,
    "TRICEP": 4,
    "BICEP": 5,
    "PECC": 6,
    "PECS": 7,
    "ADELT": 8,
    "MDELT": 9,
    "EDC": 10,
    "ECU": 11,
    "ECRB": 12
} 

muscle_mvc_max_arr = []
for muscle, col in muscles.items():
    # TODO Step 2.
    # Create path to muscle trial 1
    # Read muscle MVC file
    # process muscle
    # get muscle max for trial1
    muscle_mvc_max = -1
    for attempt in [1,2]: 
        # path mvc trial
        mvc_trial_path = f"{mvc_folder}\{sub_num}_MVC_{muscle}_{attempt}_a.tsv" 
        # reads the csv for the trial
        mvc_df = pd.read_csv(mvc_trial_path, sep='\t', header=13)
        # obtains the indexed signal of the 2nd trial
        mvc_df_signal = mvc_df[f"EMG {col}"].to_numpy()
        processed_mvc_signal = emg_sp.process_muscle(mvc_df_signal)
        # gets max value from processed signal
        mvc_trial_max = np.max(processed_mvc_signal)
        muscle_mvc_max = max(muscle_mvc_max, mvc_trial_max)
    muscle_mvc_max_arr.append(muscle_mvc_max)

    for condition, norm_maxs in normalized_maxs.items():
        folder_prefix = f'd_{sub_num}_{condition}*.tsv'
        condition_trial_paths = glob.glob(f'{trial_folder}/{folder_prefix}')
        condition_max = -1
        
        for condition_trial in condition_trial_paths:
            condition_df = pd.read_csv(condition_trial, sep='\t', header=13)
            condition_signal = condition_df[f"EMG {col}"].to_numpy()
            processed_signal = emg_sp.process_muscle(condition_signal)
            condition_trial_max = np.max(processed_signal)
            condition_max = max(condition_max, condition_trial_max) #takes the max value of the condition between prec 

        condition_maxs[condition].append(condition_max)
        normalized_max = (condition_max/muscle_mvc_max) if condition_max != -1 else -1
        norm_maxs.append(normalized_max)
        

#export mvc maxes to csv 
mvc_muscle_max_arr = np.array(muscle_mvc_max_arr)
mvc_muscle_max_arr = np.expand_dims(mvc_muscle_max_arr, axis=0)
mvc_muscle_csv = pd.DataFrame(data=mvc_muscle_max_arr, columns=muscles.keys())
mvc_muscle_csv.to_csv(f"{mvc_folder}/{sub_num}_MVC_values.csv")

df = pd.DataFrame.from_dict(condition_maxs, orient='index', columns=muscles.keys())
df.to_csv(f'{trial_folder}/{sub_num}_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_condition_maxs.csv'")

df = pd.DataFrame.from_dict(normalized_maxs, orient='index', columns=muscles.keys())
df.to_csv(f'{trial_folder}/{sub_num}_normalized_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_normalized_condition_maxs.csv'")
