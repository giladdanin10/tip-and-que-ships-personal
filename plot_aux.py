import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


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
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        plt.xticks(rotation=45, ha='right', fontsize=8)

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
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right', fontsize=8)

    # Use MaxNLocator to reduce the number of ticks
    # ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.legend(lines, labels)
    plt.title('Multiple Columns with Separate Y-Axes')
    fig.tight_layout()

    plt.show()
