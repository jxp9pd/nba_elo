"""
Code for generating visualizations using matplotlib.

Conventions for functions:
    - takes two parameters (plot_df/data and param_dict).
    - Running the function will produce the plot and return the ax object.
    - plot_df: implies dataframe
    - data: implies list of values.

Conventions for input data:
    - In a 2D plot, x value column is labeled x_val, y-value column is labeled y_val.

"""

# --------------------------------------------------------------------------------------------------
# Imports and Constants
# --------------------------------------------------------------------------------------------------

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from cycler import cycler

rh_style = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.prop_cycle': cycler('color', [(0.0, 0.784, 0.02), (0.592, 0.592, 0.592),
                                        (0.765, 0.961, 0.235)])
}


# --------------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------------

def axis_format(ax, param_dict):
    """Adds labels for axes and other standard formatting"""

    ax.set_ylabel(param_dict.get('ylabel', ''))
    ax.set_xlabel(param_dict.get('xlabel', ''))
    ax.set_title(param_dict.get('title', ''))
    ax.yaxis.set_major_formatter(param_dict.get('format_code', ''))
    ax.tick_params(axis='x', rotation=param_dict.get('rotate_degree', 0))
    if 'xlim' in param_dict:
        ax.set_xlim(param_dict['xlim'][0], param_dict['xlim'][1])
    if 'ylim' in param_dict:
        print("Hello World")
        ax.set_ylim(param_dict['ylim'][0], param_dict['ylim'][1])
    return ax


def clip_last_val(ax, clip_amt=1):
    """Clip the last value of a line plot, assumes x-axis is int"""

    curr = ax.get_xlim()
    new_lim = int(curr[1] - clip_amt)
    ax.set_xlim(curr[0], new_lim)
    return ax


# --------------------------------------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------------------------------------

# Histogram
def single_histogram(data, param_dict):
    """A helper function to make a histogram."""

    fig, ax = plt.subplots()
    ax.hist(data, bins=param_dict['bins'],
            histtype='bar', edgecolor='black')
    ax.set_title(param_dict['title'])
    ax.set_xlabel(param_dict['xlabel'])
    return fig, ax


def plot_np_histogram(plot_df, param_dict):
    """Plots a histogram of user sign-ups"""

    hist_y, bin_ticks = np.histogram(plot_df.y_val.values,
                                     bins=param_dict.get('n_bins', param_dict.get('bins', 10)))
    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        if param_dict.get('plot_line', False):
            ax.plot(bin_ticks[:-1], hist_y, color='black')
        ax.bar(bin_ticks[:-1], hist_y, width=(0.8 * bin_ticks[1]), edgecolor='black')
        ax = axis_format(ax, param_dict)
    return fig, ax


# Line Graph
def line_graph(plot_df, param_dict):
    """Plots a single line graph"""

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        ax.plot(plot_df['x_val'], plot_df['y_val'])
        ax = axis_format(ax, param_dict)
        return fig, ax


# Multi Line Graph
def multi_line_graph(plot_dfs, param_dict):
    """Plots a multi line graph"""

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        for index, plot_df in enumerate(plot_dfs):
            ax.plot(plot_df['x_val'], plot_df['y_val'], label=param_dict.get('labels', [])[index],
                    color=param_dict.get('colors', [])[index])
        ax = axis_format(ax, param_dict)
        ax.legend()
        return fig, ax


# Vertical Bar Chart
def bar_chart(data, param_dict):
    """Plots a vertical bar chart"""

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        ax.bar(param_dict['labels'], data, width=param_dict.get('bar_width', 0.8))
        ax = axis_format(ax, param_dict)
        return fig, ax


# Horizontal Bar Chart
def add_hbar_labels(x, y):
    for i in range(len(x)):
        plt.text(y[i] + 0.01, x[i], f'{y[i]:.1%}', color='black')


def h_bar_pct(data, param_dict):
    """Horizontal bar chart, where values are % of total."""

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        y_pos = np.arange(len(param_dict['labels']))
        pct_sizes = [x / param_dict.get('total_count', np.sum(data)) for x in data]
        ax.barh(y_pos, pct_sizes, align='center')
        plt.yticks(y_pos, param_dict['labels'])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(param_dict.get('title', ''), loc='left')
        add_hbar_labels(y_pos, pct_sizes)
        plt.figure(figsize=(8, 6))
        return fig, ax


# Grouped Vertical Bar Chart
def grouped_bar_chart(data1, data2, param_dict):
    """Grouped vertical bar chart."""

    bar_width = param_dict.get('bar_width', 0.4)
    r1 = np.arange(len(param_dict['labels']))
    r2 = [x + bar_width for x in r1]

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()
        ax.bar(r1, data1, width=bar_width, edgecolor='black',
               label=param_dict.get('bar1_label', ''))
        ax.bar(r2, data2, color='gray', edgecolor='black', width=bar_width,
               label=param_dict.get('bar2_label', ''))
        ax = axis_format(ax, param_dict)
        plt.xticks([r + 0.5 * bar_width for r in range(len(param_dict['labels']))
                    ], param_dict['labels'])
        ax.legend()
        return fig, ax


# Boxplot
def boxplot_comparison(plot_df, param_dict):
    """
    Boxplot Comparison.

    Parameters
    ----------
    plot_df: dataframe with column 'category' and 'val'.
    param_dict: dictionary with optional plot arguments
    Assumes user wants one boxplot per value in category.
    """

    # Pre-processing
    plot_df = plot_df.groupby('category').apply(lambda x: pd.to_numeric(
        x['val']).tolist()).reset_index()
    box_arrays = plot_df[0].tolist()
    pos = np.arange(len(box_arrays)) + 1

    with mpl.rc_context(rh_style):
        fig, ax = plt.subplots()

        ax.boxplot(box_arrays, positions=pos, sym=param_dict.get('flier_code', 'b+'),
                   labels=plot_df.category, showfliers=param_dict.get('showfliers', False))
        ax = axis_format(ax, param_dict)
        return fig, ax
