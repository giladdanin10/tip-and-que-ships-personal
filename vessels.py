from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import copy
import numpy as np
import os
from file_aux import *
from time_aux import *
from df_aux import *
import convert_aux as CONVERT
import geopandas as gpd
from shapely.geometry import Point




class VESSELES:
    # Class attribute
    vehicle_count = 0

    # Initializer / Instance attributes
    def __init__(self):
        self.data_dic = []  # It should be self.kuku to be an instance attribute
        self.info_df = []
        self.prob_MMSI = [];

    def load_data(self, input_csv_file_name_full, columns_list_keep, min_date=None, max_date=None, reload=False):
        # Convert the Time column from 'YYYYMMDD_HHMMSS' to 'YYYY-MM-DD HH:MM:SS'
        pkl_file_name_full = input_csv_file_name_full.replace(".csv", ".pkl")

        df = load_or_create_df(input_csv_file_name_full, pkl_file_name_full, reload=reload)

# in case the format has already changed
        try:
            df = convert_time_format(df, 'Time', '%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S')
            # df = CONVERT.convert_to_float(df, 'Time', '%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S')
        except:
            pass

        df = filter_df_by_date(df, min_date, max_date)

        # Get a list of interesting columns
        df = df[columns_list_keep]

        print(f'df has {df.shape[0]} lines./ncolumns are:{df.columns.to_list()}')
        return df

    def create_data_dic(self,df):
        print('creating data_dic')
        grouped = df.groupby('MMSI')

        # Create a dictionary to store each vessel's data
        self.data_dic = {MMSI: group for MMSI, group in grouped}
        return (self.data_dic)





    def create_info_df(self, min_data_len_thresh=2,to_print = True,num_lines = None):
        print('create info_df')

        self.info_df = pd.DataFrame()
        prob_MMSI = []
        MMSI_list = list(self.data_dic.keys())

        if (num_lines != None):
            MMSI_list = MMSI_list[:num_lines]



        for i, vessel_MMSI in enumerate(MMSI_list):
            if (i % 1000 == 0):
                print(f'processing MMSI {i} out of {len(MMSI_list)}')
            vessel_data = self.get_vessel_data(vessel_MMSI)  # Assuming get_vessel_data is defined elsewhere
            
            if (vessel_data.shape[0] < min_data_len_thresh):
                prob_MMSI.append(vessel_MMSI)
            else:
                vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[vessel_MMSI])
                self.info_df = pd.concat([self.info_df, vessels_df_line])

        self.info_df = self.info_df.sort_values(by='len', ascending=False)

        if (to_print):
            print (f"total number of MMSI:{len(MMSI_list)}")
            print (f"{self.info_df.shape[0]} MMSI's passed")
            print (f"{len(prob_MMSI)} MMSI's failed")


        self.prob_MMSI = prob_MMSI

        return self.info_df,prob_MMSI  # Corrected return statement


    def get_info_df_summary(self):
        info_df_summary = {}

        for column in (vessels_info_df.columns):
            info_df_summary[column] = (self.info_df[column].min(),self.info_df[column].max())

        print_dict(info_df_summary)

        return 


    def get_vessel_data(self,vessel_MMSI,to_print=False):
        
        if (1):
        # try:
            vessel_data = self.data_dic[vessel_MMSI]
            # repeat missing ID values
            ID_columns = ['IMO','Vessel_Name','Ship_Type']
            for ID_column in ID_columns:
                data = vessel_data[ID_column].loc[vessel_data[ID_column].notna()]
                data = data.unique()
                if (isinstance(data,str)):
                    data = data.strip()
                # print(vessel_data.shape)
                vessel_data = repeat_single_value_in_column(vessel_data,data,ID_column)
                # print(vessel_data.shape)
                
                if (vessel_data.empty):
                    if (to_print is True):
                        print(f'failed to create data base for vessel_MMSI={vessel_MMSI}')

                    return vessel_data


            # take only the lines where there is a Longitude
            # print(vessel_data.shape)

            vessel_data = vessel_data[vessel_data['Longitude'].notna()]

    # handle exponent represntations 
            vessel_data.loc[:, 'Latitude'] = vessel_data['Latitude'].apply(convert_to_float)
            vessel_data.loc[:, 'Longitude'] = vessel_data['Longitude'].apply(convert_to_float)
            
            # print(vessel_data.shape)
    
            # sort data by time
            vessel_data = vessel_data.sort_values(by='Time')
            # print('sucess')
        # except:
        #     vessel_data = pd.DataFrame()
        #     # sys.exit(1)   
        return vessel_data


    def get_vessel_data_stats(self,vessel_data):
        stats_dic = {
            'len': [vessel_data.shape[0]],  # Scalar value wrapped in a list
            'min_time':get_min_max_dates(vessel_data)[0],
            'max_time':get_min_max_dates(vessel_data)[1],
            'total_time':max(vessel_data['Time'])- min(vessel_data['Time']),
            'min_time_diff[mins]': round(np.min(time_diff_convert(vessel_data['Time'].diff()))),
            'max_time_diff[mins]': round(np.max(time_diff_convert(vessel_data['Time'].diff()))),
            'mean_time_diff[mins]': round(np.mean(time_diff_convert(vessel_data['Time'].diff()))),
            'min_Longitude':(min(vessel_data['Longitude'])),
            'max_Longitude':(max(vessel_data['Longitude'])),
            'min_Latitude':(min(vessel_data['Latitude'])),
            'max_Latitude':(max(vessel_data['Latitude'])),
        }
        stats_dic['span_Longitude']  = stats_dic['max_Longitude']-stats_dic['min_Longitude']
        stats_dic['span_Latitude']  = stats_dic['max_Latitude']-stats_dic['min_Latitude']

            # 'diff_Latitude':max(vessel_data['Latitude'])-min(vessel_data['Latitude'])


        return stats_dic



    def save_vessel_data_to_geojson(self,vessel_data, file_path = './data', file_name=None):
        """
        Save latitude and longitude data from a DataFrame to a GeoJSON file.
        
        Parameters:
        - vessel_data: pandas DataFrame containing 'Latitude' and 'Longitude' columns.
        - file_path: Directory path where the GeoJSON file will be saved.
        - file_name: Name of the GeoJSON file (without the .geojson extension).
        """

        if (file_name is None):
            file_name = vessel_data['Vessel_Name'].iloc[0]

        file_name = file_name.rstrip()

        # Create a geometry column with Point objects
        geometry = [Point(lon, lat) for lon, lat in zip(vessel_data['Longitude'], vessel_data['Latitude'])]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(vessel_data, geometry=geometry, crs='EPSG:4326')  # Assuming WGS84 projection

        # Save to GeoJSON file
        file_name_geojson = f"{file_path}/{file_name}.geojson"
        gdf.to_file(file_name_geojson, driver='GeoJSON')


    def save_vessels_data_to_geojson(vessels_df_info,vessel_data_dic,file_path):
        for i in range(inf_df.shape[0]):
            try:
                if (i % 10==0):
                    print(f'saving {i} files out of {inf_df.shape[0]}')
                vessel_data = get_vessel_data(vessel_data_dic,inf_df.index[i])
                save_vessel_data_to_geojson(vessel_data,file_path)
            except:
                print(f'could not export MMSI={inf_df.index[i]} to jason')
        print(f'saved {i} files in {file_path}')
        return   
