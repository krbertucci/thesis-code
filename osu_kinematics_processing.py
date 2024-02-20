# OSU PROCESSING - Kayla Russell-Bertucci ()
# INPUTS = raw calibration trials and task files
# PURPOSE
# 1. rotates to ISB
# 2. Recreates markers using LCS
# 3. Filters and pads data
# OUTPUT = Processed task trials (individual trial ranges and averaged ranges)

import pandas as pd
import numpy as np
import glob

# input subject
sub_num = "S05"
# set path for files
trial_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_Kinematics/Digitized_TSV"

# sample rate
fs = 100

# CALIBRATION

# set cal file folder
cal_file = f"{trial_folder}\CAL_1.tsv"
# reads csv into a data frame
cal_raw = pd.read_csv(cal_file, sep='\t', header = 12)
# sets the row for the cal trial (note: frame#-1)
cal_frame = 3
# create dictionary containing marker names and their columns

#create a dictionary for the markers, marker name : indexed lines from the file
markers = {
    "mcp2" : cal_raw.iloc[cal_frame, 0:2].values,
    "mcp5" : cal_raw.iloc[cal_frame,3:5].values,
    "rs" : cal_raw.iloc[cal_frame,6:8].values,
    "us" : cal_raw.iloc[cal_frame,9:11].values,
    "le" : cal_raw.iloc[cal_frame, 21:23].values,
    "me" : cal_raw.iloc[cal_frame, 24:26].values,
    "r_acr" : cal_raw.iloc[cal_frame, 36:38].values,
    "ss" : cal_raw.iloc[cal_frame, 42:44].values,
    "xp" : cal_raw.iloc[cal_frame, 60:62].values,
    "c7" : cal_raw.iloc[cal_frame, 39:41].values,
    "l_acr" : cal_raw.iloc[cal_frame, 63:65].values,
    }
# create a dictionary for the clusters. cluster marker : indexed frame from file
clusters = {
    "fa1": cal_raw.iloc[cal_frame, 12:14].values,
    "fa2" : cal_raw.iloc[cal_frame, 15:17].values,
    "fa3" : cal_raw.iloc[cal_frame, 18:20].values,
    "ua1" : cal_raw.iloc[cal_frame, 27:29].values,
    "ua2" : cal_raw.iloc[cal_frame, 30:32].values,
    "ua3" : cal_raw.iloc[cal_frame, 33:35].values,
    "chest1" : cal_raw.iloc[cal_frame, 45:47].values,
    "chest2" : cal_raw.iloc[cal_frame, 48:50].values,
    "chest3" : cal_raw.iloc[cal_frame, 51:53].values,
    "chest4" : cal_raw.iloc[cal_frame, 54:56].values,
    "chest5" : cal_raw.iloc[cal_frame, 57:59].values,
}

# Define Cal markers
# mcp2_cal = cal_raw.iloc[cal_frame,
