
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import copy
import numpy as np
import os
from display_aux import *
from vessels import VESSELES
from ships import SHIPS

from df_aux import *
from time_aux import *
from file_aux import *


def explore_mmsi(df):
    groups = df.groupby('mmsi')
    mmsi_dic = {mmsi:group for mmsi,group in groups}
    mmsi_summary_dic = {}
    mmsi_summary_dic['good_name'] = []
    mmsi_summary_dic['nan_name'] = []
    mmsi_summary_dic['multiple_name'] = []

    for i,mmsi in enumerate(mmsi_dic.keys()):
        name = mmsi_dic[mmsi]['name'].unique()
        if name.shape[0]>1:
            mmsi_summary_dic['multiple_name'].append(mmsi)
        elif pd.isna(name):
            mmsi_summary_dic['nan_name'].append(mmsi)
        else:
            mmsi_summary_dic['good_name'].append(mmsi)    

    return mmsi_summary_dic


def main_ships():
    params = {}
    params['input_csv_file_name_full'] = 'C:\\gilad\\work\\tip_and_que\\data\\AIS\\Combined\\Spire_AIS_Combined_All_20231101_1106_000000000000.csv'
    # params['input_csv_file_name_full'] = 'debug_data_base.csv'

    params['columns_list_keep'] = None
    params['filter_vessels_df_dic'] = {
            'max_time_diff[mins]':['<=',30]
            }
    params['reload_level'] = 2
    params['reload_df_filt'] = False
    params['reload_vessels'] = True
    params['save_folder_base'] = './pkl'

    params['export_to_excel'] = False
    params['ana_vessel_name'] = 'EYVAN'
    params['df_filter_dic'] = {}
    



    ships = SHIPS()
    save_folder = params['save_folder_base']+'/'+ get_file_base_name(params['input_csv_file_name_full'])
    df = ships.load_raw_data(params['input_csv_file_name_full'],reload_level=params['reload_level'],save_folder=save_folder)

    # filter_df
    df = ships.filter_df(df,reload_level=params['reload_level'],df_filter_dic=params['df_filter_dic'],save_folder=save_folder,columns_list_keep=params['columns_list_keep'])

    mmsi_summary_dic = explore_mmsi(df)
    mmsi = mmsi_summary_dic['good_name'][1]

    df_filt = df[df['mmsi']==mmsi]
    df_filt = df.iloc[range(1000)]
    df_filt = df_filt.sort_values(by='position_timestamp').reset_index(drop=True)
    diff_time = df_filt['position_timestamp'].diff().dt.total_seconds()

    plt.figure()
    plt.plot(diff_time)
    plt.show()


    data_dic = ships.create_data_dic(df,reload_level=params['reload_level'],save_folder=save_folder)






def main_vessles():
    params = {}
    params['input_csv_file_name_full'] = 'C:\\gilad\\work\\tip_and_que\\data\\AIS\\TipandCue_DataSample_CSV\\exactEarth_historical_data_02_2023.csv'
    params['input_csv_file_name_full'] = 'C:\\gilad\\work\\tip_and_que\\data\\AIS\\Combined\\Spire_AIS_Combined_All_20231101_1106_000000000000.csv'
    # params['input_csv_file_name_full'] = 'debug_data_base.csv'

    params['min_date'] = None
    params['max_date'] = None
    params['columns_list_keep'] = ['Time','MMSI','IMO','Vessel_Name','Ship_Type','Longitude','Latitude','Message_ID','Accuracy','Heading','COG','Fixing_device','Destination_ID','offset1','Offset_2','Offset_3','Offset_4','ATON_type','ATON_name','GNSS_status']
    params['filter_vessels_df_dic'] = {
            'max_time_diff[mins]':['<=',30]
            }
    params['reload_level'] = 3
    params['reload_df_filt'] = False
    params['reload_vessels'] = True
    params['save_folder_base'] = './pkl'

    params['export_to_excel'] = False
    params['ana_vessel_name'] = 'EYVAN'

    vessels = VESSELES()

    save_folder = params['save_folder_base']+'/'+ get_file_base_name(params['input_csv_file_name_full'])

    # df_org = vessels.load_data(params['input_csv_file_name_full'],columns_list_keep=params['columns_list_keep'],reload=params['reload'],remove_problematic_parts=False)
    df,status = vessels.load_raw_data(params['input_csv_file_name_full'],reload_level=params['reload_level'],save_folder=save_folder)

    # filter_df
    df,status = vessels.filter_df(df,reload_level=params['reload_level'],save_folder=save_folder,columns_list_keep=params['columns_list_keep'])

    data_dic,status = vessels.create_data_dic(df,reload_level=params['reload_level'],save_folder=save_folder)



         
    k = 1


if __name__ == '__main__':
    # main_vessles()
    main_ships()

    