
import os
import pandas as pd

def load_or_create_df(csv_file_path, save_path,reload = False):
    if os.path.exists(save_path) and reload==False:
        print(f"Loading DataFrame from {save_path}")
        df = pd.read_pickle(save_path)
    else:
        print(f"Reading CSV file from {csv_file_path}")
        df = pd.read_csv(csv_file_path, low_memory=False)
        print(f"Saving DataFrame to {save_path}")
        df.to_pickle(save_path)
    return df



