#OSU TASK EMG - Kayla Russell-Bertucci (last modified: Jan 25, 2023)
# INPUTS
#   = 1. Subject Number
#   = 2. Folder names containing MVC files and trial files
# NOTE: check that files are in the correct folders with the correct names before running 
# PURPOSE
#   filters, removes bias, & full wave rectify for each muscle of task and MVC trials
#   normalize amplitude to peak MVC
# OUTPUT 
#   = 1. MVC max values for each muscle exports to a csv
#   = 2. %MVC peak for individual muscles in each mouse sensitivity and difficulty per participant exported to csv
#   = 3. %MVC mean of each muscle in each condition. takes mean of 3 trials for each condition.
# TRIAL NAMING CONVENTIONS
# eh - easy diff high sens 
# el - easy diff low sens
# ep - easy diff pref sens
# hh - hard diff high sens
# hl - hard diff low sens
# hp  - hard diff high sens

import pandas as pd
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import emg_signal_processing_functions as emg_sp
from tqdm import tqdm

# CHANGE SUBJECT NUMBER BEFORE RUNNING SCRIPT
sub_num = "S01"
#sets path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files" 
#sets path for mvc files
mvc_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/MVC"

# dictionary with each condition and empty values - the values will be updated with the maxes from line 100
condition_maxs = {
    "EASY_PREF": [],
    "EASY_HIGH": [],
    "EASY_LOW": [],
    "HARD_PREF": [],
    "HARD_HIGH": [],
    "HARD_LOW": [],
}
# dictionary with each condition and empty values - values will be the means of each condition from line 
condition_means = {
    "EASY_PREF": [],
    "EASY_HIGH": [],
    "EASY_LOW": [],
    "HARD_PREF": [],
    "HARD_HIGH": [],
    "HARD_LOW": [],
}
# dictionary containing each condition with empty max values - max values will be updated in line 88
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

mvc_val_dict = {
    "UTRAP": [],
    "SUPRA": [],
    "INFRA": [],
    "TRICEP": [],
    "BICEP": [],
    "PECC": [],
    "PECS": [],
    "ADELT": [],
    "MDELT": [],
    "EDC": [],
    "ECU": [],
    "ECRB": []
}



def plot_signal(signal: np.array, path: str, title: str):
    """ Plot the signal.
    
    Arg:
        signal: the signal to plot.
        path: Path to save plot.
        title: the title of the plot
    """
    # Increase figure size
    plt.figure(figsize=(12,6))
    plt.plot(signal)
    plt.title(title)

    #Create path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{title}.png")
    plt.close()


# empty array to append individual mvc values in column order
muscle_mvc_max_arr = []
for muscle, col in tqdm(muscles.items()):
    # sets an mvc to -1 to be overwritten when max is calculated, or to indicate a trial is missing
    muscle_mvc_max = -1
    # iterates between trials 1 and 2, indicated by the suffix of the trial
    # NOTE: THIS WILL NOT WORK ON TRIALS WITH '3' OR '2A', RENAME FILES PRIOR TO RUNNING THE SCRIPT
    for attempt in [1,2]: 
        # path mvc trial
        mvc_trial_path = f"{mvc_folder}\{sub_num}_MVC_{muscle}_{attempt}_a.tsv" 
        # reads the csv for the trial
        mvc_df = pd.read_csv(mvc_trial_path, sep='\t', header=13)
        # obtains the indexed signal of the 2nd trial
        mvc_df_signal = mvc_df[f"EMG {col}"].to_numpy()
        # processes the signal 
        processed_mvc_signal = emg_sp.process_signal(mvc_df_signal)
        # PLOTING
        plot_signal(processed_mvc_signal, f"{trial_folder}/MVC_trials/", f"MVC_{muscle}_{attempt}_a")
        # gets max value from processed signal
        mvc_trial_max = np.max(processed_mvc_signal)
        # temporarily stores the max value between previous or current trial. previous can be muscle_mvc_max
        muscle_mvc_max = max(muscle_mvc_max, mvc_trial_max)
    # once max values are stored to muscle_mvc_max, they are appended to muscle_mvc_max_arr before the loop moves on to the next muscle 
    muscle_mvc_max_arr.append(muscle_mvc_max)
    
    # iterates through dictionary 'normalized_maxes' by condition which is paired to norm_maxs
    for condition1, norm_maxs in normalized_maxs.items():
        # sets folder prefix to obtain files
        folder_prefix = f'd_{sub_num}_{condition1}*.tsv'
        # groups trial paths based on their folder prefix which contains the condition name
        condition_trial_paths = glob.glob(f'{trial_folder}/{folder_prefix}')
        if len(condition_trial_paths) == 0: continue
        # sets max to -1. max will always be greater than -1. if csv contains -1, there is an issue or trial does not exist
        condition_max = -1
        condition_mean = []
        # iterates through trial count (condition_trial) of the grouped files in condition_trial_paths 
        for condition_trial in condition_trial_paths:
            #pulls original file name to be used for plot output
            condition_trial_basename = os.path.basename(condition_trial).strip('.tsv')
            # creates a data frame from csv containing trial data
            condition_df = pd.read_csv(condition_trial, sep='\t', header=13)
            # converts condition_df csv to a numpy file
            condition_signal = condition_df[f"EMG {col}"].to_numpy()
            # process trial signal using signal processing function (bandpass, fwr)
            processed_signal = emg_sp.process_signal(condition_signal)
            clipped_processed_signal = processed_signal(range(6000,106460))
            #plots processed signal of each condition trial
            plot_signal(clipped_processed_signal, f"{trial_folder}/Trial_EMG_plots/{muscle}", f'{condition_trial_basename}_{muscle}')
            # obtains maximal value from the processed signal
            condition_trial_max = np.max(clipped_processed_signal)
            # temporarily stores the max of the current condiion 
            condition_max = max(condition_max, condition_trial_max)  
            # obtains mean of the signal from the processed signal
            condition_trial_mean = np.mean(clipped_processed_signal)
            # appends the mean of the current trial to condition_mean
            condition_mean.append(condition_trial_mean)
        # appends the max value from condition_max to the empty value in condition_maxs dictionary
        condition_meanofmeans = np.mean(condition_mean)
        # appends the max of the condition to condition_max
        condition_maxs[condition1].append(condition_max)
        # normalize the condition mean to % MVC. if value does not exist then places -1
        normalized_mean = (condition_meanofmeans/muscle_mvc_max) if condition_meanofmeans >= 0 else -1
        # appends the normalized means to the value in the conditions_means dictionary
        condition_means[condition1].append(normalized_mean)
        # normalize the condiion maxes to the mvc max value of each muscle if the condition max is not -1
        normalized_max = (condition_max/muscle_mvc_max) if condition_max != -1 else -1
        # appends the normalized max to the value in normalized_max dictionary
        norm_maxs.append(normalized_max)

# EXPORT VALUES TO CSV
#  create an array that contains the mvc max values
mvc_muscle_max_arr = np.array(muscle_mvc_max_arr)
# adds a column of zeroes
mvc_muscle_max_arr = np.expand_dims(mvc_muscle_max_arr, axis=0)
# converts the mvc_muscle_max_arr to a dataframe containing the muscles as column names
mvc_muscle_csv = pd.DataFrame(data=mvc_muscle_max_arr, columns=muscles.keys())
# converts the data frame to a csv format
mvc_muscle_csv.to_csv(f"{mvc_folder}/{sub_num}_MVC_values.csv")

def jmp_sort(dictionary: dict, file_path: str):
    for key, value in dictionary.items():
        difficulty, sensitivity = key.split("_")
        dictionary[key] = [sub_num, difficulty, sensitivity] + value
    result_df = pd.DataFrame.from_dict(dictionary, orient='index', columns=(['Subject', 'Difficulty', 'Sensitivity'] + list(muscles.keys())))
    result_df.reset_index(drop=True)
    result_df.to_csv(file_path)

jmp_sort(normalized_maxs, f'{trial_folder}/{sub_num}_sorted_normalized_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_sorted_normalized_condition_maxs.csv")
jmp_sort(condition_means, f'{trial_folder}/{sub_num}_sorted_normalized_condition_means.csv')
print(f"subject mean emg values have been written to {trial_folder}/{sub_num}_sorted_normalized_condition_means.csv")
jmp_sort(condition_maxs, f'{trial_folder}/{sub_num}_sorted_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_sorted_condition_maxs.csv")
