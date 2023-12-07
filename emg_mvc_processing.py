import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

#STEPS
#set path
#set subject name 
#concatonate subject name with MVC trial
#set muscles to columns
#match MVC trial name to column number
#output maximum value for specific muscle
#take maximum value from multiple MVC attempts
#save max values in a file

# set subject number
sub_num = "S03"
subject_folder = r"C:\Users\kruss\OneDrive - University of Waterloo\Documents\OSU\Data\S03\Data_Raw\Trial_EMG\MVC" #sets path for files
subject_dirs = os.listdir(subject_folder) # creates a list of the files in the subject folder

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
# Filter files
# Filter Steps -> 4th order, zero lag, bandpass butterworth 
# 1. Remove DC Bias
# 2. Bandpass butterworth 
# 3. Full Wave Rectify 

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
    fs = 2000 #Sampling frequency
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
    # plot_signal(unbiased_signal, "Unbiased Signal")
    bandpass_signal = butter_bandpass(unbiased_signal)
    # plot_signal(bandpass_signal, "Bandpass Signal")
    rectified_signal = full_wave_rectify(bandpass_signal)
    # plot_signal(rectified_signal, "Rectified Signal")
    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    # ax1.plot(signal)
    # ax2.plot(unbiased_signal)
    # ax3.plot(bandpass_signal)
    # ax4.plot(rectified_signal)
    # plt.show()
    
    return rectified_signal

# def plot_signal(signal: np.ndarray, title: str):
#     plt.plot(signal)
#     plt.title(title)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
muscle_max_arr = []
for muscle, col in muscles.items(): #idx is indexing the muscle column value
    print("****** NEW MUSCLE ******")
    print(f"Processing muscle: {muscle} from column {col}")
    print(f"Processing Trial 1 ...")
    trial1_path = f"{subject_folder}\{sub_num}_MVC_{muscle}_1_a.tsv"
    t1 = pd.read_csv(trial1_path, sep='\t', header=13) # reads the trial from the indexed column
    trial1_signal = t1[f"EMG {col}"].to_numpy() #obtains the indexed signal of the 2nd trial
    processed_muscle_signal_t1 = process_muscle(trial1_signal)
    trial1_max = np.max(processed_muscle_signal_t1) #gets max value from processed trial 1 signal
    print(trial1_max)
    print(f"Processing Trial 2 ...")
    trial2_path = f"{subject_folder}/{sub_num}_MVC_{muscle}_2_a.tsv"
    t2 = pd.read_csv(trial2_path, sep='\t', header=13)
    trial2_signal = t2[f"EMG {col}"].to_numpy() #obtains the indexed signal of the 2nd trial
    processed_muscle_signal_t2 = process_muscle(trial2_signal)
    trial2_max = np.max(processed_muscle_signal_t2) #gets max value from processed trial 2 signal
    print(trial2_max)
   
    print(muscle_max_arr)
    muscle_mvc_max = max(trial1_max, trial2_max)
    print(muscle_mvc_max)
    muscle_max_arr.append(muscle_mvc_max)
    print(muscle_max_arr)
 
# write the muscle_max_arr to a csv
# convert csv to dataframe
subject_mvc_csv = pd.DataFrame(muscle_max_arr) #, columns = ['UTRAP', 'SUPRA', 'INFRA', 'TRICEP', 'BICEP', 'PECC' 'PECS', 'ADELT', 'MDELT', 'EDC', 'ECU', 'ECRB']).to_numpy()

# Transpose the MVC max csv so values can be column matched
subject_mvc_csv_t = subject_mvc_csv.T
print(subject_mvc_csv_t)

# convert max value array to csv
subject_mvc_csv_t.to_csv(f"{subject_folder}/{sub_num}_MVC_values.csv")
print(f"subject MVC values have been written to {subject_folder}/{sub_num}")
