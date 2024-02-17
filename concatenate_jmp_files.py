import pandas as pd

# Create empty list to store dataframes
subject_dfs = [] 
# Create list of 0 to 34
for sub_num in ["S05", "S05", "S05"]:
    # Create path to the csv based on list from 0 to 34
    # S{sub_num:02d}
    sub_path = f"C:/Users/kruss/OneDrive - University of Waterloo/Documents/OSU/Data/{sub_num}/Data_Raw/Trial_EMG/Trial_EMG_Files/{sub_num}_sorted_normalized_condition_means.csv"
    # Import csv into df from the path
    sub_df = pd.read_csv(sub_path, sep = ',')
    # Store the df into the list for the dataframes
    subject_dfs.append(sub_df)

concatenated_subs = pd.concat(subject_dfs, ignore_index=True)
print(concatenated_subs)
