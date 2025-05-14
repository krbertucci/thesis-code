#OSU TASK EMG - Kayla Russell-Bertucci (last modified: Mar 15, 2024)
# INPUTS
#   = 1. Subject Number
#   = 2. Folder path names containing MVC files and trial files

# NOTE: check that files are in the correct folders with the correct names before running 
# PURPOSE
#   filters, removes bias, & full wave rectify for each muscle of task and MVC trials
#   normalize amplitude to peak MVC
# OUTPUT 
#   = 1. MVC max values for each muscle exports to a csv
#   = 2. %MVC peak for individual muscles in each mouse sensitivity and difficulty per participant exported to csv
#   = 3. %MVC mean of each muscle in each condition. takes mean of 3 trials for each condition.


import pandas as pd
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import emg_signal_processing_functions as emg_sp
from tqdm import tqdm

# CHANGE SUBJECT NUMBER BEFORE RUNNING SCRIPT
sub_num = "S25"
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
    # iterates between trials 1 and 2, indicated by the suffix of the trial, then proccesses, plots and stores
    # NOTE: THIS WILL NOT WORK ON TRIALS WITH '3' OR '2A', RENAME FILES PRIOR TO RUNNING THE SCRIPT
    for attempt in [1,2]: 
        mvc_trial_path = f"{mvc_folder}\{sub_num}_MVC_{muscle}_{attempt}_a.tsv"
        mvc_df = pd.read_csv(mvc_trial_path, sep='\t', header=13)
        mvc_df_signal = mvc_df[f"EMG {col}"].to_numpy()
        processed_mvc_signal = emg_sp.process_signal(mvc_df_signal)
        plot_signal(processed_mvc_signal, f"{trial_folder}/MVC_trials/", f"MVC_{muscle}_{attempt}_a")
        mvc_trial_max = np.max(processed_mvc_signal)
        # temporarily stores the max value between previous or current trial.
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
        condition_max = -1
        condition_mean = []
        for condition_trial in condition_trial_paths:
            #pulls original file name to be used for plot output
            condition_trial_basename = os.path.basename(condition_trial).strip('.tsv')
            condition_df = pd.read_csv(condition_trial, sep='\t', header=13)
            condition_signal = condition_df[f"EMG {col}"].to_numpy()
            processed_signal = emg_sp.process_signal(condition_signal)
            #clips outer 4.5 seconds off each trial to remove processing noise
            clipped_processed_signal = processed_signal[9000:107000]
            plot_signal(clipped_processed_signal, f"{trial_folder}/Trial_EMG_plots/", f'{muscle}_{condition_trial_basename}')
            # obtain max and mean values from the processed signal
            condition_trial_max = np.max(clipped_processed_signal) 
            condition_max = max(condition_max, condition_trial_max)  
            condition_trial_mean = np.mean(clipped_processed_signal)
            condition_mean.append(condition_trial_mean)
        # append the max and mean vals for each condition to respective dictionarys  dictionary
        condition_meanofmeans = np.mean(condition_mean)
        condition_maxs[condition1].append(condition_max)
        normalized_mean = (condition_meanofmeans/muscle_mvc_max) if condition_meanofmeans >= 0 else -1
        condition_means[condition1].append(normalized_mean)
        normalized_max = (condition_max/muscle_mvc_max) if condition_max != -1 else -1
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

def jmp_transform(dictionary: dict, file_path: str):
    for key, value in dictionary.items():
        difficulty, sensitivity = key.split("_")
        dictionary[key] = [sub_num, difficulty, sensitivity] + value
    result_df = pd.DataFrame.from_dict(dictionary, orient='index', columns=(['Subject', 'Difficulty', 'Sensitivity'] + list(muscles.keys())))
    result_df.reset_index(drop=True)
    result_df.to_csv(file_path)

jmp_transform(normalized_maxs, f'{trial_folder}/{sub_num}_sorted_normalized_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_sorted_normalized_condition_maxs.csv")
jmp_transform(condition_means, f'{trial_folder}/{sub_num}_sorted_normalized_condition_means.csv')
print(f"subject mean emg values have been written to {trial_folder}/{sub_num}_sorted_normalized_condition_means.csv")
jmp_transform(condition_maxs, f'{trial_folder}/{sub_num}_sorted_condition_maxs.csv')
print(f"subject max emg values have been written to {trial_folder}/{sub_num}_sorted_condition_maxs.csv")
