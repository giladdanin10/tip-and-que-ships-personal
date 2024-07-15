
import numpy as np
import pandas as pd
from convert_aux import * 
import convert_aux as CONVERT

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
        subset_df.to_excel(out_file_name, index=False)
        
    elif (file_extension=='.csv'):
        subset_df.to_csv(out_file_name, index=False)


