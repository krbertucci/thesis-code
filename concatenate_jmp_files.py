import pandas as pd

data_folder = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/Data_Analysis"

# Create empty list to store dataframes
subject_means_dfs = []
subject_maxs_dfs = [] 
# Create list of 0 to 34
    # change range to the desired subjects
for sub_num in range(1,35):
    # Create path to the csv based on list from 0 to 34
    # S{sub_num:02d} adds a 0 to single digit subject numbers
    sub_path_means = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/S{sub_num:02d}/Data_Raw/Trial_EMG/Trial_EMG_Files/S{sub_num:02d}_sorted_normalized_condition_means.csv"
    sub_path_maxs = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/S{sub_num:02d}/Data_Raw/Trial_EMG/Trial_EMG_Files/S{sub_num:02d}_sorted_normalized_condition_maxs.csv"
    # Import csv into df from the path
    sub_means_df = pd.read_csv(sub_path_means, sep = ',')
    sub_maxs_df = pd.read_csv(sub_path_maxs, sep = ',')
    # Store the df into the list for the dataframes
    subject_means_dfs.append(sub_means_df)
    subject_maxs_dfs.append(sub_maxs_df)

concatenated_subs_means = pd.concat(subject_means_dfs, ignore_index=True)
concatenated_subs_maxs = pd.concat(subject_maxs_dfs, ignore_index=True)
#print(concatenated_subs_means)
#print(concatenated_subs_maxs)
concatenated_subs_means.to_csv(f"{data_folder}/osu_subject_means.csv")
concatenated_subs_maxs.to_csv(f"{data_folder}/osu_subject_maxs.csv")
print(f'files have been saved to {data_folder}')