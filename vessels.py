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
import sys
from IPython.display import display


class VESSELES:
    # Class attribute
    vehicle_count = 0

    # Initializer / Instance attributes
    def __init__(self):
        self.data_dic = []  # It should be self.kuku to be an instance attribute
        self.info_df = []
        self.prob_MMSI = []
        self.prob_dic = {}
        self.lines_removed = []

    def load_raw_data(self, input_csv_file_name_full, reload_level=0,save_folder = './pkl'):
        print ('load_row_data')
        print ('----------------')

        df = None 
        status = True
        # Convert the Time column from 'YYYYMMDD_HHMMSS' to 'YYYY-MM-DD HH:MM:SS'
        
        pkl_file_name_full = save_folder + '/' + 'df_org.pkl'

# basic df 
        if (reload_level==0) or not (os.path.exists(pkl_file_name_full)) :
            print(f"Reading CSV file from {input_csv_file_name_full}")
            df = pd.read_csv(input_csv_file_name_full, low_memory=False)
            status = save_var(df,pkl_file_name_full,'df')

        elif (reload_level==1):
            df = load_df_from_file(pkl_file_name_full)

        else:
            df = None  
        return df,status
    

    def filter_df(self,df,reload_level=0,save_folder = './pkl',min_date=None, max_date=None,columns_list_keep=None):
        print ('filter_df')
        print ('----------------')
        status = True
        df = None
        df_filt_file_name = save_folder + '/df_filt.pkl'

# filtered df
        if (reload_level>2) and self.check_pkl_files_exists(save_folder):
            return df,status
        
        elif (reload_level>=2) and os.path.exists(df_filt_file_name):
            df,status = load_df_from_file(df_filt_file_name)
            return df,status
        else:
            if (df is None):
                df_org_file_name = save_folder + '/df_org.pkl'
                df,status = load_df_from_file(df_org_file_name)
                
    # in case the format has already changed
            try:
                df = convert_time_format(df, 'Time', '%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S')
                # df = CONVERT.convert_to_float(df, 'Time', '%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S')
            except:
                pass

    # trim spaces from Vessel_name
            df['Vessel_Name'] = df['Vessel_Name'].apply(lambda x: str(x).strip() if isinstance(x, str) else x)

            
            df = filter_df_by_date(df, min_date=None, max_date=None)

            # Get a list of interesting columns
            if (columns_list_keep != None):
                df = df[columns_list_keep]


    # add the original index
            df['org_index'] = df.index


    # convert numerical data to floats
            float_numerical_columns = ['Longitude','Latitude']
            for column in float_numerical_columns:
                df[column] = df[column].apply(convert_to_float)


    # remove bad MMSIs
        grouped = df.groupby('MMSI')
        data_dic = {MMSI: group for MMSI, group in grouped}    
        no_vessel_name_list = []
        multiple_vessel_name_list = []
        single_vessel_name_list = []

        # detect MMSIs with none or multiple vessel_names
        for i,MMSI in enumerate(data_dic.keys()):
            if (i % 1000 == 0):
                print(f'processing MMSI {i} out of {len(data_dic.keys())}')
            # if (data_dic[MMSI]['Vessel_Name'][data_dic[MMSI]['Vessel_Name'].notna()].shape[0] != 0):
            Vessel_name = data_dic[MMSI]['Vessel_Name'][data_dic[MMSI]['Vessel_Name'].notna()].unique()
            if (Vessel_name.shape[0]==0):
                no_vessel_name_list.append(MMSI)
            elif (Vessel_name.shape[0]==1):
                single_vessel_name_list.append(MMSI)
                df.loc[data_dic[MMSI].index,'Vessel_Name']=Vessel_name[0]


            elif (Vessel_name.shape[0]>1):
                multiple_vessel_name_list.append(MMSI)

        self.prob_dic['MMSI'] = {'none':no_vessel_name_list,'mutiple':multiple_vessel_name_list}                

        filter_dic = {'MMSI':['==',single_vessel_name_list]}
        df = filter_df(df,filter_dic)  




        status = save_var(df,file_name=df_filt_file_name,var_name='df')              

        return df,status

    def check_pkl_files_exists(self,save_folder = './pkl'):
        data_dic_file_name = save_folder+'/data_dic.pkl'
        info_df_file_name = save_folder+'/info_df.pkl'
        prob_dic_file_name = save_folder+'/prob_dic.pkl'
        
        if ( (os.path.exists(data_dic_file_name)) and (os.path.exists(data_dic_file_name)) and (os.path.exists(data_dic_file_name)) ):
            return True
        else:
            return False

                                                     

    def create_data_dic(self,df,vessel_names = None,vessel_data_length_thresh=5,save_folder = './pkl',reload_level=0):
        print ('create_data_dic')
        print ('----------------')

        data_dic_file_name = save_folder+'/data_dic.pkl'
        info_df_file_name = save_folder+'/info_df.pkl'
        prob_dic_file_name = save_folder+'/prob_dic.pkl'
        status = True

        

        if (reload_level>=3) and self.check_pkl_files_exists(save_folder):   
            self.data_dic,status = load_var(data_dic_file_name)
            self.info_df_dic,status = load_var(info_df_file_name)
            self.prob_dic,status = load_var(prob_dic_file_name)
            return self.data_dic,status

        else:
            df_filt_file_name = save_folder + '/df_filt.pkl'
            if (df is None):
                df,status = load_df_from_file(df_filt_file_name)

            self.info_df = pd.DataFrame()
    # prepare prob_dic
            self.prob_dic['Vessel_Name'] = {}

            ID_columns = ['IMO','Ship_Type']
            for ID_column in ID_columns:
                self.prob_dic['Vessel_Name'][ID_column] = {'none':[],'multiple':[]}

            self.prob_dic['Vessel_Name']['short'] = []

    # group by Vessel_Name
            grouped = df.groupby('Vessel_Name')
            self.data_dic = {Vessel_Name: group for Vessel_Name, group in grouped}    


            good_list = []
    # make complete missing data at lines containing nans at certain colums
            if (vessel_names is None):
                vessel_names = self.data_dic.keys()
                
            for i,Vessel_Name in enumerate(vessel_names):
                if (i % 100 == 0):
                    print(f'processing Vessel_Name {i} out of {len(self.data_dic.keys())}')

                vessel_data = self.data_dic[Vessel_Name]

    # prepare the vessel_data                            
                self.data_dic[Vessel_Name],status = self.prepare_vessel_data(vessel_data,ID_columns=ID_columns)            
                
                if (self.data_dic[Vessel_Name].shape[0]<vessel_data_length_thresh):
                    self.prob_dic['Vessel_Name']['short'].append(Vessel_Name)
                    continue
    # get some statistics             
                vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
                self.info_df = pd.concat([self.info_df,vessels_df_line])

            save_var(self.data_dic,data_dic_file_name)            
            save_var(self.info_df,info_df_file_name)  
            save_var(self.prob_dic,prob_dic_file_name)                    

            # # remove_problematic_parts if desired
            # if (remove_problematic_parts):
            #     filter_dic = {'Vessel_Name':['==',good_list]}
            #     df = filter_df(df,filter_dic)  

            
            return df,status



#     def pre_process_Vessel_Name_based(self,df,remove_problematic_parts=True,to_print=False,vessel_names=None,vessel_data_length_thresh=5,out_folder = './',reload=False):
#         data_dic_file_name = out_folder+'data_dic.pkl'
#         info_df_file_name = out_folder+'info_df.pkl'
#         prob_dic_file_name = out_folder+'prob_dic.pkl'

        
#         if not reload and ((os.path.exists(data_dic_file_name) and os.path.exists(info_df_file_name))):
#             self.data_dic = load_var(data_dic_file_name)
#             self.info_df_dic = load_var(info_df_file_name)
#             self.prob_dic = load_var(prob_dic_file_name)

#             return        


#         self.info_df = pd.DataFrame()
#         print('pre_process_Vessel_Name_based')
#         print('------------------------------')

# # prepare prob_dic
#         self.prob_dic['Vessel_Name'] = {}

#         ID_columns = ['IMO','Ship_Type']
#         for ID_column in ID_columns:
#             self.prob_dic['Vessel_Name'][ID_column] = {'none':[],'multiple':[]}

#         self.prob_dic['Vessel_Name']['short'] = []
# # group by Vessel_Name
#         grouped = df.groupby('Vessel_Name')
#         self.data_dic = {Vessel_Name: group for Vessel_Name, group in grouped}    

    
    

#         good_list = []
# # make complete missing data at lines containing nans at certain colums
#         if (vessel_names is None):
#             vessel_names = self.data_dic.keys()
            
#         for i,Vessel_Name in enumerate(vessel_names):
#             if (i % 100 == 0):
#                 print(f'processing Vessel_Name {i} out of {len(self.data_dic.keys())}')

#             vessel_data = self.data_dic[Vessel_Name]

# # prepare the vessel_data
                        
#             self.data_dic[Vessel_Name],status = self.prepare_vessel_data(vessel_data,ID_columns=ID_columns,to_print=to_print)            
            
#             if (self.data_dic[Vessel_Name].shape[0]<vessel_data_length_thresh):
#                 self.prob_dic['Vessel_Name']['short'].append(Vessel_Name)
#                 continue
# # get some statistics             
#             vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
#             self.info_df = pd.concat([self.info_df,vessels_df_line])

#         save_var(self.data_dic,data_dic_file_name)            
#         save_var(self.info_df,info_df_file_name)  
#         save_var(self.prob_dic,prob_dic_file_name)                    

#         # # remove_problematic_parts if desired
#         # if (remove_problematic_parts):
#         #     filter_dic = {'Vessel_Name':['==',good_list]}
#         #     df = filter_df(df,filter_dic)  

        
#         return df

    def prepare_vessel_data(self,vessel_data,ID_columns = ['IMO','Ship_Type'],to_print=False):
        lines_remove = []
        Vessel_Name = vessel_data['Vessel_Name'].iloc[0]
        if (to_print):
            print(f'prepare_vessel_data for {Vessel_Name}')
        
# deal with ID_colums
        for ID_column in ID_columns:
            data = vessel_data[ID_column].loc[vessel_data[ID_column].notna()]
            data = data.unique()
            if (isinstance(data,str)):
                data = data.strip()
            # print(vessel_data.shape)

            if (data.shape[0]==0):
                self.prob_dic['Vessel_Name'][ID_column]['none'].append(Vessel_Name)
                return vessel_data,False
                
            elif (data.shape[0]==1):
                try:
                    vessel_data.loc[vessel_data.index,ID_column] = data[0]
                    
                except Exception as e:
                    print(f"Vessel_Name:{Vessel_Name} ID_cloumn:{ID_column} Data:{data}")                   
                    # Print the exception message
                    print(f"An error occurred: {e}")
                    # Exit the program
                    sys.exit(1)
                                        
            elif (data.shape[0] > 1):
                self.prob_dic['Vessel_Name'][ID_column]['multiple'].append(Vessel_Name)

        
        # find lines with duplicate time stamp


        
        vessel_data = vessel_data.sort_values(by='Time')
        vessel_data = vessel_data.reset_index(drop=True)

    
        zero_diff_line_numbers = vessel_data[vessel_data['Time'].diff()==pd.Timedelta(0)].index

        # for each duplicants create a combined line and place it in the 
        for line_number in zero_diff_line_numbers:
            check_df = vessel_data.loc[[line_number-1,line_number]]
            combined_row = check_df.iloc[1].combine_first(check_df.iloc[0])



            # vessel_data.loc[zero_diff_line_numbers[0]-1] = combined_row
            vessel_data.loc[line_number] = combined_row

        # keep a reocrd of the removed line
            # self.lines_remove.append(vessel_data['org_index'].loc[line_number-1])

        # remove the 1'st line of the pair
            vessel_data = vessel_data.drop(index=line_number-1)


        # return to the original index    
        vessel_data = vessel_data.set_index('org_index')
        

        
# remove lines with Latitude = Nan        
        vessel_data = vessel_data.loc[vessel_data['Latitude'].notna()]
        
        # self.removed_lines.append(vessel_data[vessel_data['Latitude'].isna()].index)
        return vessel_data,True

    # def create_data_dic(self,df):
    #     print('creating data_dic')
    #     # grouped = df.groupby('MMSI')

    #     # # Create a dictionary to store each vessel's data
    #     # self.data_dic = {MMSI: group for MMSI, group in grouped}
    #     # return (self.data_dic)
    #     grouped = df.groupby('MMSI')
    #     data_dic = {MMSI: group for MMSI, group in grouped}    
    #     no_vessel_name_list = []
    #     multiple_vessel_name_list = []
    #     single_vessel_name_list = []

    #     # detect MMSIs with none or multiple vessel_names
    #     for i,MMSI in enumerate(data_dic.keys()):
    #         if (i % 1000 == 0):
    #             print(f'processing MMSI {i} out of {len(data_dic.keys())}')
    #         data_dic[MMSI]['Vessel_Name']
    #         # if (data_dic[MMSI]['Vessel_Name'][data_dic[MMSI]['Vessel_Name'].notna()].shape[0] != 0):
    #         Vessel_name = data_dic[MMSI]['Vessel_Name'][data_dic[MMSI]['Vessel_Name'].notna()].unique()
    #         if (Vessel_name.shape[0]==0):
    #             no_vessel_name_list.append(MMSI)

    #         elif (Vessel_name.shape[0]==1):
    #             single_vessel_name_list.append(MMSI)
    #             df.loc[data_dic[MMSI].index,'Vessel_Name']=Vessel_name[0]


    #         elif (Vessel_name.shape[0]>1):
    #             multiple_vessel_name_list.append(MMSI)

    #     # # filter them out 
    #     filter_dic = {'MMSI':['==',single_vessel_name_list]}
    #     df = filter_df(df,filter_dic)  

    #     # # create the data_dic this time with Vesssel_name

    #     grouped = df.groupby('Vessel_Name')


    #     self.data_dic = {Vessel_name: group for Vessel_name, group in grouped}
    #     self.prob_dic['MMSI'] = {'no_vessel_name':no_vessel_name_list,'multiple_vessel_name':multiple_vessel_name_list}






    # def create_info_df(self, min_data_len_thresh=2,to_print = True,num_lines = None):
    #     print('create info_df')

    #     self.info_df = pd.DataFrame()
    #     prob_MMSI = []
    #     MMSI_list = list(self.data_dic.keys())

    #     if (num_lines != None):
    #         MMSI_list = MMSI_list[:num_lines]



    #     for i, vessel_MMSI in enumerate(MMSI_list):
    #         if (i % 1000 == 0):
    #             print(f'processing MMSI {i} out of {len(MMSI_list)}')
    #         vessel_data = self.get_vessel_data(vessel_MMSI)  # Assuming get_vessel_data is defined elsewhere

    #         if (vessel_data.shape[0] < min_data_len_thresh):
    #             prob_MMSI.append(vessel_MMSI)
    #         else:
    #             Vessel_Name = vessel_data['Vessel_Name'].iloc[0]
    #             vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
    #             self.info_df = pd.concat([self.info_df, vessels_df_line])

    #     self.info_df = self.info_df.sort_values(by='len', ascending=False)

    #     if (to_print):
    #         print (f"total number of MMSI:{len(MMSI_list)}")
    #         print (f"{self.info_df.shape[0]} MMSI's passed")
    #         print (f"{len(prob_MMSI)} MMSI's failed")


    #     self.prob_MMSI = prob_MMSI

    #     return self.info_df,prob_MMSI  # Corrected return statement

    def create_info_df(self, df,min_data_len_thresh=2,to_print = True,num_lines = None):
        print('create info_df')

        self.info_df = pd.DataFrame()
        prob_Vessel_Name = []
        Vessel_Name_list = df['Vessel_Name'].unique()

        if (num_lines != None):
            Vessel_Name_list = Vessel_Name_list[:num_lines]



        for i, vessel_Name in enumerate(Vessel_Name_list):
            if (i % 1000 == 0):
                print(f'processing Vessel_Name {i} out of {len(Vessel_Name_list)}')
            vessel_data = self.get_vessel_data(df,vessel_Name)  # Assuming get_vessel_data is defined elsewhere

            if (vessel_data.shape[0] < min_data_len_thresh):
                prob_Vessel_Name.append(vessel_Name)
            else:
                Vessel_Name = vessel_data['Vessel_Name'].iloc[0]
                try:
                    vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(df,Vessel_Name),index=[Vessel_Name])
                    
                except Exception as e:
                    # Print the exception message
                    print(vessel_Name)
                    print(f"An error occurred: {e}")
                    # Optionally, you can log the error or perform other actions here
                    # Exit the program
                    sys.exit(1)


                self.info_df = pd.concat([self.info_df, vessels_df_line])

        self.info_df = self.info_df.sort_values(by='len', ascending=False)

        if (to_print):
            print (f"total number of Vessel_Name:{len(Vessel_Name_list)}")
            print (f"{self.info_df.shape[0]} Vessel_Name's passed")
            print (f"{len(prob_Vessel_Name)} Vessel_Name's failed")


        self.prob_Vessel_Name = prob_Vessel_Name

        return self.info_df,prob_Vessel_Name  # Corrected return statement


    def create_info_df1(self, df,min_data_len_thresh=2,to_print = True,num_lines = None):
        print('create info_df')

        self.info_df = pd.DataFrame()
        prob_Vessel_Name = []
        Vessel_Name_list = df['Vessel_Name'].unique()

        if (num_lines != None):
            Vessel_Name_list = Vessel_Name_list[:num_lines]



        for i, vessel_Name in enumerate(Vessel_Name_list):
            if (i % 1000 == 0):
                print(f'processing Vessel_Name {i} out of {len(Vessel_Name_list)}')
            vessel_data = self.get_vessel_data(df,vessel_Name)  # Assuming get_vessel_data is defined elsewhere

            if (vessel_data.shape[0] < min_data_len_thresh):
                prob_Vessel_Name.append(vessel_Name)
            else:
                Vessel_Name = vessel_data['Vessel_Name'].iloc[0]
                try:
                    vessels_df_line = pd.DataFrame(self.get_vessel_data_stats1(vessel_data),index=[Vessel_Name])
                    
                except Exception as e:
                    # Print the exception message
                    print(vessel_Name)
                    print(f"An error occurred: {e}")
                    # Optionally, you can log the error or perform other actions here
                    # Exit the program
                    sys.exit(1)







                self.info_df = pd.concat([self.info_df, vessels_df_line])

        self.info_df = self.info_df.sort_values(by='len', ascending=False)

        if (to_print):
            print (f"total number of Vessel_Name:{len(Vessel_Name_list)}")
            print (f"{self.info_df.shape[0]} Vessel_Name's passed")
            print (f"{len(prob_Vessel_Name)} Vessel_Name's failed")


        self.prob_Vessel_Name = prob_Vessel_Name

        return self.info_df,prob_Vessel_Name  # Corrected return statement




    def get_info_df_summary(self):
        info_df_summary = {}

        for column in (vessels_info_df.columns):
            info_df_summary[column] = (self.info_df[column].min(),self.info_df[column].max())

        print_dict(info_df_summary)

        return 





    def get_vessel_data(self,df,Vessel_Name,to_print=False):
    
        vessel_data = df.loc[df['Vessel_Name']==Vessel_Name]
        vessel_data = vessel_data[vessel_data['Longitude'].notna()]
    
        # sort data by time
        vessel_data = vessel_data.sort_values(by='Time')
        return vessel_data


    # def get_vessel_data_stats(self,df,vessel_name):
    #     vessel_data = self.get_vessel_data(df,vessel_name)

    #     stats_dic = {
    #         'len': [vessel_data.shape[0]],  # Scalar value wrapped in a list
    #         'min_time':get_min_max_dates(vessel_data)[0],
    #         'max_time':get_min_max_dates(vessel_data)[1],
    #         'total_time':max(vessel_data['Time'])- min(vessel_data['Time']),
    #         'min_time_diff[mins]': round(np.min(time_diff_convert(vessel_data['Time'].diff()))),
    #         'max_time_diff[mins]': round(np.max(time_diff_convert(vessel_data['Time'].diff()))),
    #         'mean_time_diff[mins]': round(np.mean(time_diff_convert(vessel_data['Time'].diff()))),
    #         'min_Longitude':(min(vessel_data['Longitude'])),
    #         'max_Longitude':(max(vessel_data['Longitude'])),
    #         'min_Latitude':(min(vessel_data['Latitude'])),
    #         'max_Latitude':(max(vessel_data['Latitude'])),
    #     }
    #     stats_dic['span_Longitude']  = stats_dic['max_Longitude']-stats_dic['min_Longitude']
    #     stats_dic['span_Latitude']  = stats_dic['max_Latitude']-stats_dic['min_Latitude']

    #         # 'diff_Latitude':max(vessel_data['Latitude'])-min(vessel_data['Latitude'])


    #     return stats_dic

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
