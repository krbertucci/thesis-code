# file reorganization for JMP 
import pandas as pd
import numpy as np
import os


# first column = subject number
# second column = indicate level difficukty (EASY VS HARD)
# third column = will indicate mouse sensitivity (HIGH LOW PREF)
# columns 4 onwards will alternate between muscle mean and muscle peak (UTRAP mean | UTRAP max)

sub_num = "S01"
#import files containing condition maxes and means


con_means_file = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files/{sub_num}_normalized_condition_means"
con_maxes_file = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files/{sub_num}_normalized_condition_maxs"

con_means_df = pd.read_csv(con_means_file, sep=',', header=1)
con_maxes_df = pd.read_csv(con_maxes_file, sep=',', header=1)

