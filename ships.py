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
from parse_aux import *
from plot_aux import *
from str_aux import *




class SHIPS:
    # Class attribute
    vehicle_count = 0

    # Initializer / Instance attributes
    def __init__(self,save_folder = './pkl'):
        self.data_dic = []  # It should be self.kuku to be an instance attribute
        self.info_df = []
        self.lines_removed = []
        self.save_folder = save_folder
        self.fresh_data_set = True        

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
        return status
    

    def prepare_df(self,df,reload_level=0,save_folder = './pkl',df_filter_dic={},columns_list_keep=None):
        print ('prepare_df')
        print ('----------------')


        status = True
        df = None
        df_filt_file_name = save_folder + '/df_filt.pkl'

# filtered df        
        if (reload_level>=2) and os.path.exists(df_filt_file_name):
            df,status = load_df_from_file(df_filt_file_name)
            self.fresh_data_set = False
            return df
        else:
            if (df is None):
                df_org_file_name = save_folder + '/df_org.pkl'
                df,status = load_df_from_file(df_org_file_name)
                

            # filter the df
            df = filter_df(df,df_filter_dic)

            # trim spaces from Vessel_name
            df['name'] = df['name'].apply(lambda x: str(x).strip() if isinstance(x, str) else x)

            # make sure the time related columns are indeed in datetime format
            df['position_timestamp'] = pd.to_datetime(df['position_timestamp'])
            df['static_timestamp'] = pd.to_datetime(df['static_timestamp'])

            # chnage df['name'] = Nan to 'Nan' 
            ind_name_nan = df['name'].isna()
            df.loc[ind_name_nan,'name'] = 'Nan'

            # add time columm as the position_timestamp 
            # df.rename(columns={'position_timestamp': 'time'}, inplace=True)
            df['time'] = df['position_timestamp']

            # add numeric time column
            df['time_seconds'] = df['time'].astype('int64')/10**9


            # Get a list of interesting columns
            if (columns_list_keep != None):
                df = df[columns_list_keep]

            df = reorder_df_columns(df,['time','time_seconds','name','imo','mmsi','latitude','longitude','position_timestamp','static_timestamp'])


    # add the original index
            df['org_index'] = df.index


    # convert numerical data to floats
            float_numerical_columns = ['longitude','latitude']
            for column in float_numerical_columns:
                df[column] = df[column].apply(convert_to_float)

    # modify ship names
            df = self.modify_ship_names(df)
            status = save_var(df,file_name=df_filt_file_name,var_name='df')              

        return df

    def modify_ship_names(self,df):
        df['name'] = df['name'].apply(lambda x: replace_char_with_count(x, '@'))
        return  df


    def check_pkl_files_exists(self,save_folder = './pkl'):
        # data_dic_file_name = save_folder+'/data_dic.pkl'
        info_df_file_name = save_folder+'/info_df.pkl'
        # prob_dic_file_name = save_folder+'/prob_dic.pkl'
        
        if ( (os.path.exists(info_df_file_name)) ):
            return True
        else:
            return False



# data_structure can be either a df or a dictionary of dfs
    def get_ship_df(self,ship_name,**params):
        default_params = {
            'item_type': {'default': 'name', 'optional': {'name','mmsi'}},
            'sort_columns':{'default':'time'},
            'handle_common_time_rows': {'default': True},
            'modify_data_dic': {'default': True},
            'reprocess':False
        }

        params = parse_func_params(params, default_params)

        
        ship_df = self.data_dic[ship_name]

        if (params['sort_columns'] is not None):
            parse_parameter(params['sort_columns'],ship_df.columns)        

            ship_df = ship_df.sort_values(by=params['sort_columns'])



        if (not self.info_df.loc[ship_name,'processed']) or (params['reprocess']):

            if (params['handle_common_time_rows']):
                ship_df = handle_common_time_rows_in_df(ship_df,ID_columns=params['item_type'],time_column='static_timestamp')

# remnove lines with Nan in Latitude
            ship_df = ship_df.loc[ship_df['latitude'].notna()]

# remove bias from 
            # ship_df['time_seconds_no_bias'] = ship_df['time_seconds']-ship_df['time_seconds'].min()

            if (params['modify_data_dic']):          
                self.data_dic[ship_name] = ship_df
                self.info_df.loc[ship_name,'processed'] = True

        return ship_df



    # def get_ship_data_stats(ship_data,item_type=None,id_column_check=[]):
    def get_ship_data_stats(self,ship_data,**params):
        default_params = {
            'item_type': {'default': 'name', 'optional': {'name','mmsi'}},
            'id_column_check': {'default': []}
    }

        try:
            params = parse_func_params(params, default_params)
        except ValueError as e:
            print(e)  # Print the exception message with calling stack path
            return None

        
        if (not isinstance(params['id_column_check'],list)):
            params['id_column_check'] = [params['id_column_check']]

        if (ship_data.shape[0]==1):
            stats_dic = {
            params['item_type']:ship_data[params['item_type']].unique().tolist(),
            'len': [ship_data.shape[0]],  # Scalar value wrapped in a list
            }
        else:

            stats_dic = {
                params['item_type']:ship_data[params['item_type']].unique().tolist(),
                'len': [ship_data.shape[0]],  # Scalar value wrapped in a list
                'min_time':min(ship_data['time']),
                'max_time':max(ship_data['time']),
                'total_time':max(ship_data['time'])- min(ship_data['time']),
                'min_time_diff[mins]': round(np.min(time_diff_convert(ship_data['time'].diff()))),
                'max_time_diff[mins]': round(np.max(time_diff_convert(ship_data['time'].diff()))),
                'mean_time_diff[mins]': round(np.mean(time_diff_convert(ship_data['time'].diff()))),
                'min_longitude':(min(ship_data['longitude'])),
                'max_longitude':(max(ship_data['longitude'])),
                'min_latitude':(min(ship_data['latitude'])),
                'max_latitude':(max(ship_data['latitude'])),
            }
            stats_dic['span_longitude']  = stats_dic['max_longitude']-stats_dic['min_longitude']
            stats_dic['span_latitude']  = stats_dic['max_latitude']-stats_dic['min_latitude']

            for column in params['id_column_check']:
                stats_dic[f'num_{column}s'] = ship_data[column].unique().shape[0]



        return stats_dic



    # def create_info_df(df,num_lines = None,item_type='mmsi',id_column_check='name'):
    def create_info_df(self,**params):
        default_params = {
            'item_type': {'default': 'name', 'optional': {'name','mmsi'}},
            'num_lines': {'default': None},
            'id_column_check': {'default': 'mmsi'},
            'reset':False,
            'save_folder':self.save_folder
        }

        try:
            params = parse_func_params(params, default_params)
        except ValueError as e:
            print(e)  # Print the exception message with calling stack path
            return None


        if (params['reset'] or self.fresh_data_set==True):  
            self.info_df = pd.DataFrame(index=self.data_dic.keys())
            self.info_df['processed'] = False


        elif(os.path.exists(params['save_folder'] + '/info_df.pkl')):
            self.info_df,status = load_var(params['save_folder'] + '/info_df.pkl')
    
        

        

        print('create info_df')
        print('----------------')

        info_df = pd.DataFrame()
        
        item_list = list(self.data_dic.keys())


        if (params['num_lines'] != None):
            item_list = item_list[:params['num_lines']]

        # filter out those already processed
        item_list = [item for item in item_list if not self.info_df.loc[item,'processed']]

        for i, item in enumerate(item_list):
            if (i % 1000 == 0):
                print(f"processing {params['item_type']} {i} out of {len(item_list)}")
            # ship_data = get_item_df(item_dict,item,item_type=item_type)  # Assuming get_ship_data is defined elsewhere
            get_ship_df_params = params
            get_ship_df_params['strict_params'] = False 
            ship_data = self.get_ship_df(item,**get_ship_df_params)  # Assuming get_ship_data is defined elsewhere


            item = ship_data[params['item_type']].iloc[0]

            
            try:
                # ships_df_line = pd.DataFrame(get_ship_data_stats(ship_data,item_type=item_type,id_column_check=id_column_check),index=[item])
                get_ship_df_params = params
                get_ship_df_params['strict_params'] = False 
                ships_df_line = pd.DataFrame(self.get_ship_data_stats(ship_data,**get_ship_df_params),index=[item])

            except Exception as e:
                # Print the exception message
                print(item)
                print(f"An error occurred: {e}")
                # Optionally, you can log the error or perform other actions here
                # Exit the program
                sys.exit(1)


            info_df = pd.concat([info_df, ships_df_line])

# save internediate results
            if ((i % 1000 == 0) and (i>0)):
                self.info_df = merge_dataframes_on_index(self.info_df, info_df,how='left',mode='replace')
                self.info_df.sort_values(by='processed',inplace=True,ascending=False)

                save_var(self.info_df,self.save_folder + '/info_df.pkl')

# save final results
        self.info_df = merge_dataframes_on_index(self.info_df, info_df,how='left',mode='replace')
        self.info_df.sort_values(by='processed',inplace=True,ascending=False)

        save_var(self.info_df,self.save_folder + '/info_df.pkl')


        # info_df = info_df.reset_index(drop=True)
        return self.info_df

    def get_ship_stats(self,ship_name):
        ship_data = self.get_ship_df(ship_name)
        return self.get_ship_data_stats(ship_data)  
    



    def plot_ship_data(self,ship_names,**params):
        default_params = {
            'columns': {'default': ['latitude','longitude','time_seconds']},
            'x_data_type': {'default': 'index', 'optional': {'index', 'time'}},
            'marker_points': {'default': None},
            'marker_points_style': {'default': 'o', 'optional': {'o', 'x', 's', 'd', 'ro'}},
            'marker_style': {'default': None, 'optional': {None, 'o', 'x', 's', 'd'}},
            'line_style': {'default': '-', 'optional': {'-', '--', '-.', ':'}},
            'line_styles': {'default': None},  # Adding support for multiple line styles
            'x_label': {'default': 'Index'},
            'y_label': {'default': 'Value'},
            'xlim': {'default': None},
            'ylim': {'default': None},
            'title': {'default': 'Plot of Data'},
            'legend': {'default': True, 'optional': {True, False}},
            'legend_loc': {'default': 'upper right', 'optional': {'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}},
            'figsize': {'default': None},
            'color': {'default': None},
            'time_column':'time',
            'axes_size':None,
            'sort_columns': 'time',
            'pre_process_params': {'default':{}},
            'max_plot_ships': {'default': 16},
        }
        try:
            params = parse_func_params(params, default_params)
        except ValueError as e:
            print(e)  # Print the exception message with calling stack path
            return None
        
      
        

        if isinstance(ship_names,pd.Index):
            ship_names = ship_names.tolist()
        elif isinstance(ship_names,str):
            ship_names = [ship_names]

        if (params['max_plot_ships'] is not None):
            num_plot_ships = min(len(ship_names),params['max_plot_ships'])
        else:
            num_plot_ships = len(ship_names)

        if ('time' in params['columns']):
            print ('use time_seconds instead of time')
            return


        num_cols = min(4, num_plot_ships)
        num_rows = (num_plot_ships + num_cols - 1) // num_cols  # Ceiling division to ensure all axes fit


        if (params['axes_size'] is None):
            if (num_cols==1):
                params['axes_size']=(6, 6/3*2)
            elif (num_cols==2):
                params['axes_size'] = (4, 4/3*2)
            elif (num_cols==3):
                params['axes_size']=(3.5, 3.5/3*2)
            elif (num_cols==4):
                params['axes_size'] = (3, 2)

# extend the axes size if time is used
        if (params['x_data_type'] == 'time'):
            temp = (params['axes_size'][0]*1.1,params['axes_size'][1]*1.3)
            params['axes_size'] = temp

        


        fig, axes,num_rows,num_cols = create_subplot_scheme(axes_size=params['axes_size'], num_axes=num_plot_ships)


        # if (params['ylim'] is None):
        #     info_df_summry = self.get_info_df_summary()
        #     params['ylim'] = [info_df_summry['max_span_longitude']

        for i, ax in enumerate(axes):
            if i > len(ship_names):
                break

            get_ship_df_params = params
            get_ship_df_params['strict_params'] = False
            ship_df = self.get_ship_df (ship_names[i],**get_ship_df_params)


        
            columns_reg = [string for string in params['columns'] if string != 'time_seconds']

            min_values = ship_df.loc[:,columns_reg].min()
            max_values = ship_df.loc[:,columns_reg].max()

            time_span_params = {
                    'span':{'val':[min(min_values),max(max_values)],'columns':['time_seconds']},
            }
            
            params['pre_process_params'].update(time_span_params)


            plot_params = params
            plot_params['strict_params'] = False
            plot_params['ax'] = ax
            plot_params['title'] = ship_names[i]
            ax = plot_df_columns(ship_df,**plot_params)
        
# usage
# plot_ship_data(df,info_df.index[range(4)].tolist(),columns=['latitude', 'longitude'],ylim = [-90,90],pre_process='remove_bias')

































    def create_data_dic(self,df,vessel_names = None,vessel_data_length_thresh=5,save_folder = './pkl',reload_level=0):
        print ('create_data_dic')
        print ('----------------')

    #     data_dic_file_name = save_folder+'/data_dic.pkl'
    #     info_df_file_name = save_folder+'/info_df.pkl'
    #     prob_dic_file_name = save_folder+'/prob_dic.pkl'
    #     status = True

        

    #     if (reload_level>=3) and self.check_pkl_files_exists(save_folder):   
    #         self.data_dic,status = load_var(data_dic_file_name)
    #         self.info_df_dic,status = load_var(info_df_file_name)
    #         self.prob_dic,status = load_var(prob_dic_file_name)
    #         return self.data_dic,status

    #     else:
    #         df_filt_file_name = save_folder + '/df_filt.pkl'
    #         if (df is None):
    #             df,status = load_df_from_file(df_filt_file_name)

    #         self.info_df = pd.DataFrame()
    # # prepare prob_dic
    #         self.prob_dic['name'] = {}

    #         ID_columns = ['IMO','Ship_Type']
    #         for ID_column in ID_columns:
    #             self.prob_dic['name'][ID_column] = {'none':[],'multiple':[]}

    #         self.prob_dic['name']['short'] = []

    # # group by Vessel_Name
        grouped = df.groupby('name')
        self.data_dic = {Vessel_Name: group for Vessel_Name, group in grouped}    

        self.info_df = pd.DataFrame(index=self.data_dic.keys())
        self.info_df['processed'] = False

        
        return self.data_dic
    #         good_list = []
    # # make complete missing data at lines containing nans at certain colums
    #         if (vessel_names is None):
    #             vessel_names = self.data_dic.keys()
                
    #         for i,Vessel_Name in enumerate(vessel_names):
    #             if (i % 100 == 0):
    #                 print(f'processing Vessel_Name {i} out of {len(self.data_dic.keys())}')

    #             vessel_data = self.data_dic[Vessel_Name]

    # # prepare the vessel_data                            
    #             self.data_dic[Vessel_Name],status = self.prepare_vessel_data(vessel_data,ID_columns=ID_columns)            
                
    #             if (self.data_dic[Vessel_Name].shape[0]<vessel_data_length_thresh):
    #                 self.prob_dic['name']['short'].append(Vessel_Name)
    #                 continue
    # # get some statistics             
    #             vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
    #             self.info_df = pd.concat([self.info_df,vessels_df_line])

    #         save_var(self.data_dic,data_dic_file_name)            
    #         save_var(self.info_df,info_df_file_name)  
    #         save_var(self.prob_dic,prob_dic_file_name)                    

    #         # # remove_problematic_parts if desired
    #         # if (remove_problematic_parts):
    #         #     filter_dic = {'name':['==',good_list]}
    #         #     df = filter_df(df,filter_dic)  

            
    #         return status



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
#         self.prob_dic['name'] = {}

#         ID_columns = ['IMO','Ship_Type']
#         for ID_column in ID_columns:
#             self.prob_dic['name'][ID_column] = {'none':[],'multiple':[]}

#         self.prob_dic['name']['short'] = []
# # group by Vessel_Name
#         grouped = df.groupby('name')
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
#                 self.prob_dic['name']['short'].append(Vessel_Name)
#                 continue
# # get some statistics             
#             vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
#             self.info_df = pd.concat([self.info_df,vessels_df_line])

#         save_var(self.data_dic,data_dic_file_name)            
#         save_var(self.info_df,info_df_file_name)  
#         save_var(self.prob_dic,prob_dic_file_name)                    

#         # # remove_problematic_parts if desired
#         # if (remove_problematic_parts):
#         #     filter_dic = {'name':['==',good_list]}
#         #     df = filter_df(df,filter_dic)  

        
#         return df

    def prepare_vessel_data(self,vessel_data,ID_columns = ['IMO','Ship_Type'],to_print=False):
        lines_remove = []
        Vessel_Name = vessel_data['name'].iloc[0]
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
                self.prob_dic['name'][ID_column]['none'].append(Vessel_Name)
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
                self.prob_dic['name'][ID_column]['multiple'].append(Vessel_Name)

        
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
    #         data_dic[MMSI]['name']
    #         # if (data_dic[MMSI]['name'][data_dic[MMSI]['name'].notna()].shape[0] != 0):
    #         Vessel_name = data_dic[MMSI]['name'][data_dic[MMSI]['name'].notna()].unique()
    #         if (Vessel_name.shape[0]==0):
    #             no_vessel_name_list.append(MMSI)

    #         elif (Vessel_name.shape[0]==1):
    #             single_vessel_name_list.append(MMSI)
    #             df.loc[data_dic[MMSI].index,'name']=Vessel_name[0]


    #         elif (Vessel_name.shape[0]>1):
    #             multiple_vessel_name_list.append(MMSI)

    #     # # filter them out 
    #     filter_dic = {'MMSI':['==',single_vessel_name_list]}
    #     df = filter_df(df,filter_dic)  

    #     # # create the data_dic this time with Vesssel_name

    #     grouped = df.groupby('name')


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
    #             Vessel_Name = vessel_data['name'].iloc[0]
    #             vessels_df_line = pd.DataFrame(self.get_vessel_data_stats(vessel_data), index=[Vessel_Name])
    #             self.info_df = pd.concat([self.info_df, vessels_df_line])

    #     self.info_df = self.info_df.sort_values(by='len', ascending=False)

    #     if (to_print):
    #         print (f"total number of MMSI:{len(MMSI_list)}")
    #         print (f"{self.info_df.shape[0]} MMSI's passed")
    #         print (f"{len(prob_MMSI)} MMSI's failed")


    #     self.prob_MMSI = prob_MMSI

    #     return self.info_df,prob_MMSI  # Corrected return statement







    def get_vessel_data(self,df,Vessel_Name,to_print=False):
    
        vessel_data = df.loc[df['name']==Vessel_Name]
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
            file_name = vessel_data['name'].iloc[0]

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


    def get_info_df_summary(self,info_df=None):
        if (info_df is None):
            info_df = self.info_df

        info_df_stats = {
            'num_total':info_df.shape[0],
            'num_processed': len(info_df['processed']),
            'min_len': info_df.len.min(),
            'max_len': info_df.len.max(),
            'min_time': info_df.min_time.min(),
            'max_time': info_df.max_time.max(),
            'max_total_time': info_df.total_time.max(),
            'min_latitude': info_df.min_latitude.min(),
            'max_latitude': info_df.max_latitude.max(),
            'min_longitude': info_df.min_longitude.min(),
            'max_longitude': info_df.max_longitude.max(), 
            'max_span_longitude': info_df.span_longitude.max(), 
            'max_span_latitude': info_df.span_latitude.max(), 
            'min_span_longitude': info_df.span_longitude.min(), 
            'min_span_latitude': info_df.span_latitude.min(), 
            'num_rows_nan_ship_name': info_df.loc[info_df.name==''].len,
            }
        return info_df_stats
