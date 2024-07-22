
import numpy as np
import pandas as pd
from convert_aux import * 
import convert_aux as CONVERT
from time_aux import *
import os




def filter_df(df, filter_dic):
    """
    Filters a DataFrame based on a dictionary of column filters.

    Parameters:
    df (pd.DataFrame): The DataFrame to be filtered.
    filter_dic (dict): A dictionary where keys are column names and values are tuples.
                       Each tuple contains an operator as the first element and the filter value(s) as the second element.
                       Supported operators: '==', '!=', '<', '<=', '>', '>=', 'between'.

    Returns:
    pd.DataFrame: The filtered DataFrame or an empty DataFrame if any column does not exist.

    Example Usage:
    inf_df = pd.DataFrame({
        'MMSI': [123456789, 987654321, 192837465],
        'Vessel_Name': ['Vessel A', 'Vessel B', 'Vessel C'],
        'Latitude': [34.5, 45.6, 56.7],
        'Longitude': [-123.4, -134.5, -145.6]
    })

    filter_dic = {
        'Latitude': ('between', (40.0, 50.0)),  # Applying a 'between' filter for Latitude
        'Longitude': ('<=', -134.5),  # Applying a '<=' filter for Longitude
        'Vessel_Name': ('==', ['Vessel A', 'Vessel C']),  # Applying an '==' filter for Vessel_Name
        'Nonexistent_Column': ('==', 'SomeValue')  # Nonexistent column
    }

    filtered_df = filter_df(inf_df, filter_dic)
    print(filtered_df)
    """
    for column, (operator, value) in filter_dic.items():
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist in the DataFrame. Existing columns: {list(df.columns)}")
            return pd.DataFrame()  # Return an empty DataFrame
        
        if operator == '==':
            if isinstance(value, list):
                df = df[df[column].isin(value)]
            else:
                df = df[df[column] == value]
        elif operator == '!=':
            if isinstance(value, list):
                df = df[~df[column].isin(value)]
            else:
                df = df[df[column] != value]

        elif operator == '<':
            df = df[df[column] < value]
        elif operator == '<=':
            df = df[df[column] <= value]
        elif operator == '>':
            df = df[df[column] > value]
        elif operator == '>=':
            df = df[df[column] >= value]
        elif operator == 'between':
            if isinstance(value, tuple) and len(value) == 2:
                lower_bound, upper_bound = value
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            else:
                raise ValueError(f"Value for 'between' must be a tuple of two elements: {value}")
            
            if (lower_bound>upper_bound):
                print(f'lower bound ({lower_bound}) is higher than upper bound ({upper_bound})')

        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
    return df


def repeat_single_value_in_column (df,value,column_name,to_print=False):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]

    # print(value)        
    if (len(value) != 1):
        if (to_print):
            print(f'value is not unique:{value}')
        return pd.DataFrame()
    
    df[column_name] = np.repeat(value,df.shape[0])
    return df


def get_time_related_df_columns(df):
    time_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    return time_columns



def export_df(df, out_file_name, columns=None, start_line=0, num_lines=None):
    """
    Exports a subset of a DataFrame to an Excel file.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    out_file_name (str): The name of the output Excel file.
    columns (list): List of columns to include in the export.
    start_line (int): The starting line (index) from which to export.
    num_lines (int): The number of lines (rows) to export.

    Returns:
    None
    """

    if (columns==None):
        columns = df.columns

    if (num_lines==None):
        num_lines = df.shape[0]
    # Select the desired subset of the DataFrame
    subset_df = df[columns].iloc[start_line:start_line+num_lines-1]
    print(subset_df.shape)
    
    # Export the subset to an Excel file

    file_name, file_extension = os.path.splitext(out_file_name)

    print(f'exporting {num_lines} lines from df to {out_file_name}')

    if (file_extension=='.xlsx'):
        time_columns = get_time_related_df_columns(subset_df)

        for column in time_columns:
            subset_df.loc[:, column] = subset_df[column].dt.tz_localize(None)


        subset_df.to_excel(out_file_name, index=False)
        
    elif (file_extension=='.csv'):
        subset_df.to_csv(out_file_name, index=False)

def reorder_df_columns(df, order):
    """
    Reorder the columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be reordered.
    order (list): A list specifying the desired order of columns. Columns not specified in the list will be appended at the end.

    Returns:
    pd.DataFrame: A DataFrame with columns reordered as specified.
    """
    # Ensure the columns in 'order' exist in the DataFrame
    order = [col for col in order if col in df.columns]

    # Get the remaining columns that are not in the 'order' list
    remaining_cols = [col for col in df.columns if col not in order]

    # Concatenate the specified columns with the remaining columns
    new_order = order + remaining_cols

    # Reorder the DataFrame columns
    return df[new_order]




def handle_common_time_rows_in_df(df, time_column='time', ID_columns=[]):

    # handle a none list input 
    if (not isinstance(ID_columns,list)):
        ID_columns = [ID_columns]

    # Check if the specified columns exist in the DataFrame
    missing_columns = [col for col in ID_columns + [time_column] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")
    
    # Initialize a counter for the number of common time chunks found
    common_time_chunks_count = 0
    
    # Get the total number of chunks
    total_chunks = df.groupby(ID_columns).ngroups
    
    # Function to handle rows with common time values in chunks defined by ID_columns
    def combine_rows(chunk, chunk_number):
        nonlocal common_time_chunks_count
        
        # Sort the chunk by the time_column
        chunk = chunk.sort_values(by=time_column)
        
        # Calculate the time differences in seconds
        time_diff = chunk[time_column].diff()
        time_diff = time_diff_convert(time_diff, units='secs')
        zero_diff_line_numbers = np.where(time_diff == 0)[0]
        
        # Initialize a list to hold the indices of rows to be dropped
        indices_to_drop = []
        
        # Iterate over the indices with zero time differences and combine rows
        for line_number in zero_diff_line_numbers:
            # Ensure we have at least two rows to combine
            if line_number > 0:
                combined_row = chunk.iloc[line_number].combine_first(chunk.iloc[line_number - 1])
                
                # Place the combined row at the index of the first row in the group
                first_index = chunk.index[line_number - 1]
                df.loc[first_index] = combined_row
                
                # Add the index of the current row to the drop list
                indices_to_drop.append(chunk.index[line_number])
                
                common_time_chunks_count += 1  # Increment the counter
        
        # Drop the rows that have been combined
        if (len(indices_to_drop)!=0):
            df.drop(indices_to_drop, inplace=True)
            # df.loc[indices_to_drop, :] = df.drop(indices_to_drop)

            
        # Print progress every 1000 chunks
        if chunk_number % 10 == 0:
            print(f"Processed chunk {chunk_number} out of {total_chunks}")

    # Apply the function to each group defined by ID_columns
    for chunk_number, (_, chunk) in enumerate(df.groupby(ID_columns, group_keys=False), start=1):
        combine_rows(chunk, chunk_number)

    print(f"Number of common time chunks found: {common_time_chunks_count}")
    
    return df


import pandas as pd

def normalize_columns(df, columns, method='min-max', add_norm_columns=True):
    """
    Normalize specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to be normalized.
    columns (list or str): A list of column names or a single column name to normalize.
    method (str): The normalization method to use ('min-max' or 'z-score').
    add_norm_columns (bool): If True, adds normalized columns with suffix '_norm'. If False, replaces the original columns.

    Returns:
    pd.DataFrame: The DataFrame with normalized columns.
    """
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    df_normalized = df.copy()
    
    for column in columns:
        if method == 'min-max':
            norm_col = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        elif method == 'z-score':
            norm_col = (df[column] - df[column].mean()) / df[column].std()
        else:
            raise ValueError("Method must be 'min-max' or 'z-score'")
        
        if add_norm_columns:
            df_normalized[column + '_norm'] = norm_col
        else:
            df_normalized[column] = norm_col
    
    return df_normalized

# # Sample DataFrame
# data = {
#     'value1': [10, 20, 30, 40, 50],
#     'value2': [5, 15, 25, 35, 45],
#     'value3': [2, 4, 6, 8, 10]
# }
# df = pd.DataFrame(data)

# # Normalize specified columns and add normalized columns
# columns_to_normalize = ['value1', 'value2']
# method = 'min-max'
