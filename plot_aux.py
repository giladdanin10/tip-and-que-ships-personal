import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from  parse_aux import *
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from df_aux import *    


# 




# # Example usage:
# y = np.sin(np.linspace(0, 10, 100))
# plot(np.arange(100), y),
#      marker_points=np.array([10, 20, 30]), marker_points_style='ro',
#      marker_style='x', line_style='--', x_label='X-axis',
#      y_label='Y-axis', xlim=(0, 100), ylim=(-1, 1),
#      title='Sine Wave', legend=True, figsize=(10, 5), color='green')

# plot(y),
#      marker_points=np.array([10, 20, 30]), marker_points_style='ro',
#      marker_style='x', line_style='--', x_label='X-axis',
#      y_label='Y-axis', xlim=(0, 100), ylim=(-1, 1),
#      title='Sine Wave', legend=True, figsize=(10, 5), color='green')

# plot(y_data=y,marker_points=np.array([10, 20, 30]), marker_points_style='ro',
#      marker_style='x', line_style='--', x_label='X-axis',
#      y_label='Y-axis', xlim=(0, 100), ylim=(-1, 1),
#      title='Sine Wave', legend=True, figsize=(10, 5), color='green')



def plot(*args, **params):
    # Define default and optional values for each parameter in default_params
    default_params = {
        'y_data': {'default': np.array([])},
        'x_data': {'default': None},
        'marker_points': {'default': None},
        'marker_points_style': {'default': 'o', 'optional': {'o', 'x', 's', 'd', 'ro'}},
        'marker_style': {'default': None, 'optional': {None, 'o', 'x', 's', 'd'}},
        'line_style': {'default': '-', 'optional': {'-', '--', '-.', ':'}},
        'x_label': {'default': 'Index'},
        'y_label': {'default': 'Value'},
        'xlim': {'default': None},
        'ylim': {'default': None},
        'title': {'default': 'Plot of NumPy Array'},
        'legend': {'default': True, 'optional': {True, False}},
        'figsize': {'default': None},
        'color': {'default': 'blue', 'optional': {'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'}},
        'ax': {'default': None},
        'xlabel_fontsize': {'default': 10},
        'ylabel_fontsize': {'default': 10},
        'title_fontsize': {'default': 12},
        'tick_labelsize': {'default': 10}, 
        'legend_fontsize': {'default': 10},        
    }

    # Parse params
    try:
        params = parse_func_params(params, default_params)
    except ValueError as e:
        print(e)  # Print the exception message with calling stack path
        return None




    # Determine x_data and y_data from args or params
    if len(args) == 2:
        x_data, y_data = args
    elif len(args) == 1:
        y_data = args[0]
        x_data = params.get('x_data')
    else:
        y_data = params.get('y_data')
        x_data = params.get('x_data')

    # Use index for x_data if not provided
    if x_data is None:
        x_data = np.arange(len(y_data))
        time_data  = False
    else:
        
        time_data = pd.api.types.is_datetime64_any_dtype(x_data)
        x_data = np.asarray(x_data)
        if len(x_data) != len(y_data):
            raise ValueError("Length of x_data must be equal to length of y_data.")
    
    
    # Ensure y_data is a NumPy array
    y_data = np.asarray(y_data)
    
    # Create a figure and axis if ax is not provided
    if params['ax'] is None:
        fig, ax = plt.subplots(figsize=params['figsize'])
    else:
        ax = params['ax']
    


    
    # Plot the values
    line = ax.plot(x_data, y_data, params['line_style'], label='Data', marker=params['marker_style'], color=params['color'])
    
    
    if time_data:
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        # plt.xticks(rotation=45, ha='right', fontsize=8)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.tick_params(axis='x', rotation=45, labelsize=params['tick_labelsize'], labelright=True)
    else:
        ax.tick_params(axis='x',labelsize=params['tick_labelsize'], labelright=True)


    # Highlight marker_points if provided
    if params['marker_points'] is not None:
        ax.plot(x_data[params['marker_points']], y_data[params['marker_points']], params['marker_points_style'], linestyle='None', label='marker_points')
    
    
    # Set labels and title
    ax.set_xlabel(params['x_label'], fontsize=params['xlabel_fontsize'])
    ax.set_ylabel(params['y_label'], fontsize=params['ylabel_fontsize'])
    ax.set_title(params['title'], fontsize=params['title_fontsize'])
    
    # Set limits if provided
    if params['xlim'] is not None:
        ax.set_xlim(params['xlim'])
    if params['ylim'] is not None:
        ax.set_ylim(params['ylim'])
    
    # Add a legend if required
    if params['legend']:
        ax.legend(fontsize=params['legend_fontsize'])

    
    # Show the plot if a new figure was created
    if params['ax'] is None:
        plt.show()

    return line

def plot_df_columns(df, **params):
    # Define default and optional values for each parameter in default_params
    default_params = get_plot_defualt_params()

    def get_plot_defualt_params():
        default_params = {
            'y_data': {'default': np.array([])},
            'x_data': {'default': None},
            'marker_points': {'default': None},
            'marker_points_style': {'default': 'o', 'optional': {'o', 'x', 's', 'd', 'ro'}},
            'marker_style': {'default': None, 'optional': {None, 'o', 'x', 's', 'd'}},
            'line_style': {'default': '-', 'optional': {'-', '--', '-.', ':'}},
            'x_label': {'default': 'Index'},
            'y_label': {'default': 'Value'},
            'xlim': {'default': None},
            'ylim': {'default': None},
            'title': {'default': 'Plot of NumPy Array'},
            'legend': {'default': True, 'optional': {True, False}},
            'figsize': {'default': None},
            'color': {'default': 'blue', 'optional': {'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'}},
            'ax': {'default': None},
            'xlabel_fontsize': {'default': 10},
            'ylabel_fontsize': {'default': 10},
            'title_fontsize': {'default': 12},
            'tick_labelsize': {'default': 10}, 
            'legend_fontsize': {'default': 10},  
        return default_params      
    }
        

    plot_df = df.copy()
    try:
        params = parse_func_params(params, default_params)
    except ValueError as e:
        print(e)  # Print the exception message with calling stack path
        return None

    # Determine x_data based on x_data_type
    if (params['x_data_type']=='index'):
        x_data = range(df.shape[0])
        x_label = 'index'
    elif (params['x_data_type']=='time'):
        parse_parameter(params['time_column'],df.columns)
        x_data = df[params['time_column']]
        x_data = pd.to_datetime(x_data, utc=True)
        x_label = 'time'

    if (isinstance(params['columns'], str)):
        params['columns'] = [params['columns']]

    # Create a figure and axis if ax is not provided
    if params['ax'] is None:
        fig, ax = plt.subplots(figsize=params['figsize'])
    else:
        ax = params['ax']
    
    # Generate colors if not provided
    num_columns = len(params['columns'])
    if params['color'] is None:
        colors = create_color_vector(num_columns)
    else:
        if isinstance(params['color'], str):
            colors = cycle([params['color']])
        elif len(params['color']) != num_columns:
            raise ValueError("Number of colors must match the number of columns")
        else:
            colors = cycle(params['color'])
    
    # Determine line styles if provided
    if params['line_styles'] is None:
        line_styles = cycle([params['line_style']] * num_columns)
    else:
        if len(params['line_styles']) != num_columns:
            raise ValueError("Number of line styles must match the number of columns")
        else:
            line_styles = cycle(params['line_styles'])
    

    # pre_proccess
    for column in params['columns']: 
        pre_process_df_columns_params = params
        pre_process_df_columns_params['strict_params']=False  
        plot_df = pre_process_df_columns(plot_df,**pre_process_df_columns_params)
        

    # Plot each column with its own label for the legend
    handles = []
    for color, line_style, column in zip(colors, line_styles, params['columns']):
        y_data = plot_df[column]
        plot_params = {key: value for key, value in params.items() if key not in ['ax', 'columns', 'x_data_type', 'legend_loc', 'line_styles']}
        plot_params['color'] = color
        plot_params['line_style'] = line_style
        plot_params['strict_params'] = False
        handle = plot(x_data=x_data, y_data=y_data, ax=ax, **plot_params, label=column)
        handles.append(handle)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(params['y_label'])
    ax.set_title(params['title'])
    
    # Set limits if provided
    if params['xlim'] is not None:
        ax.set_xlim(params['xlim'])
    if params['ylim'] is not None:
        ax.set_ylim(params['ylim'])
    
    # Add a legend if required
    if params['legend']:
        ax.legend(params['columns'], loc=params['legend_loc'])
    
    # Show the plot if a new figure was created
    if params['ax'] is None:
        plt.show()
    
    return ax   



# plot_df_columns(ship_data, columns=['latitude', 'longitude'],
#  line_styles=['--', '-.'],
#   y_label='Values', legend=True, figsize=(10, 5),
#   color=['red', 'blue',],marker_points=[10, 20, 30],
#   marker_points_style='o',title='Ship Data',
#   ylim=(0, 100), legend_loc='upper right',x_data_type='time')

def create_color_vector(num_colors):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    color_cycle = cycle(colors)
    color_vector = [next(color_cycle) for _ in range(num_colors)]
    return color_vector


def create_subplot_scheme(axes_size=None, num_axes=1, max_axes_in_row=4):
    """
    Creates a subplot scheme and returns an array of axes.

    Parameters:
    ----------
    axes_size : tuple
        Size of each individual subplot (width, height).

    num_axes : int
        Total number of subplots to create.

    max_axes_in_row : int, optional
        Maximum number of subplots in a row. Default is 4.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    axes : array-like of matplotlib.axes.Axes
        Array of created subplot axes.
    """
    # Calculate the number of rows and columns
    num_cols = min(max_axes_in_row, num_axes)
    num_rows = (num_axes + num_cols - 1) // num_cols  # Ceiling division to ensure all axes fit

    if (axes_size is None):
        if (num_cols==1):
            axes_size=(6, 6/3*2)
        elif (num_cols==2):
            axes_size = (4, 4/3*2)
        elif (num_cols==3):
            axes_size=(3.5, 3.5/3*2)
        elif (num_cols==4):
            axes_size = (3, 2)




    # Calculate figure size based on individual axes size
    fig_width = axes_size[0] * num_cols
    fig_height = axes_size[1] * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    
    # Flatten the axes array if there are multiple rows or columns
    if num_rows * num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Hide any unused subplots
    for i in range(num_axes, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:num_axes],num_cols,num_rows



def plot_multiple_y_axes(df, columns=None, axes=None, x_column='index', fig_size=(5, 3)):
    """
    Plots multiple columns of a DataFrame with separate Y-axes on a single plot.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.

    columns : list
        List of column names in the DataFrame to be plotted on the Y-axes.

    axes : list, optional
        List of labels for the Y-axes corresponding to each column. 
        If None, will use column names as labels.

    x_column : str, optional
        Specifies the X-axis data. Options are:
        - 'index': uses the DataFrame index as the X-axis.
        - 'time': uses the 'time' column in the DataFrame as the X-axis.

    fig_size : tuple, optional
        Size of the figure. Default is (5, 3).

    Raises:
    ------
    ValueError:
        If the length of columns and axes are not the same.

    Example:
    --------
    plot_multiple_y_axes(df, columns=['col1', 'col2'], axes=['Column 1', 'Column 2'], x='time')
    """

    if axes is None:
        axes = columns

    if len(columns) != len(axes):
        raise ValueError("Length of columns and axes must be the same")

    fig, ax1 = plt.subplots(figsize=fig_size)

    color_map = plt.cm.get_cmap('tab10')
    lines = []
    labels = []

    if x_column == 'index':
        x = df.index
    else:
        x = df[x_column]

    ax1.plot(x, df[columns[0]], color=color_map(0), marker='o', label=axes[0])
    ax1.set_xlabel('Index')
    ax1.set_ylabel(axes[0], color=color_map(0))
    ax1.tick_params(axis='y', labelcolor=color_map(0))

# if it's a datetime
    # if pd.api.types.is_datetime64_any_dtype(df[x_column]):
    #     plt.xticks(rotation=45, ha='right', fontsize=8)

    lines.append(ax1.lines[-1])
    labels.append(axes[0])

    axes_list = [ax1]
    for i in range(1, len(columns)):
        ax = ax1.twinx()
        ax.spines['right'].set_position(('outward', 60 * i))
        ax.plot(x, df[columns[i]], color=color_map(i), marker='x', label=axes[i])
        ax.set_ylabel(axes[i], color=color_map(i))
        ax.tick_params(axis='y', labelcolor=color_map(i))

        lines.append(ax.lines[-1])
        labels.append(axes[i])
        axes_list.append(ax)

    # Rotate x-axis labels
# if it's a datetime
    # if pd.api.types.is_datetime64_any_dtype(df[x_column]):
    #     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    #     plt.xticks(rotation=45, ha='right', fontsize=8)

    # Use MaxNLocator to reduce the number of ticks
    # ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.legend(lines, labels)
    plt.title('Multiple Columns with Separate Y-Axes')
    fig.tight_layout()

    plt.show()

