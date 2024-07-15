
import os
import pandas as pd
import pickle


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


def load_df_from_file(file_name):
    print(f'load df from {file_name}')
    try:
        with open(file_name, 'rb') as file:
            var = pd.read_pickle(file)
        return var,True
    except Exception as e:
        print(f'could not load df from {file_name}. Error: {e}')
        return None,False


def save_var(var, file_name, var_name='var'):
    print(f'save {var_name} to {file_name}')
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        # Save the variable to the file
        with open(file_name, 'wb') as file:
            pickle.dump(var, file)
        return True
    except Exception as e:
        print(f'could not save {var_name} to {file_name}. Error: {e}')
        return False
    

def load_var(file_name, var_name='var'):
    status = True
    print(f'load {var_name} from {file_name}')
    try:
        with open(file_name, 'rb') as file:
            var = pickle.load(file)
        return var,status
    except Exception as e:
        print(f'could not load {var_name} from {file_name}. Error: {e}')
        status = False
        return None,status
    

def get_file_base_name(file_path):
    # Get the file name from the path
    file_name = os.path.basename(file_path)
    # Split the file name and extension
    file_base, _ = os.path.splitext(file_name)
    return file_base

    
