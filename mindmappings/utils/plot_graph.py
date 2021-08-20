#!env python

"""
    This function plots the graph and saves it in the provided path.

    Inputs:

    plot_x_axis
    plot_y_axis
    labels = (plot_xlabel,plot_ylabel)
    plot_labels
    path
    file_name

    To use, add this in your file:

    import sys
    sys.path.insert(0,'/shared/reusable/graphs')
    from plot_graph import plot_graph

"""

import numpy as np
import sys
import os
import matplotlib
from collections import OrderedDict
import matplotlib.ticker as mtick

COLORS = ['palevioletred', 'lightseagreen', 'slateblue', 'darkkhaki', 'darkslategrey',
          'mediumorchid', 'peru', 'limegreen', 'lightcoral', 'darkgoldenrod', 
          'firebrick', 'orangered', 'palegoldenrod', 'midnightblue','slategrey',
          'bisque', 'darkgreen', 'plum', 'deeppink', 'sienna']

MARKERS = ['^', '<', 's', 'o', '^', 'v', '>', 'p', 's', 'h', 'H']

linestyle_ref = OrderedDict(
        [('solid',               (0, ())),
              ('loosely dotted',      (0, (1, 10))),
              ('dotted',              (0, (1, 5))),
              ('densely dotted',      (0, (1, 1))),

              ('loosely dashed',      (0, (5, 10))),
              ('dashed',              (0, (5, 5))),
              ('densely dashed',      (0, (5, 1))),

              ('loosely dashdotted',  (0, (3, 10, 1, 10))),
              ('dashdotted',          (0, (3, 5, 1, 5))),
              ('densely dashdotted',  (0, (3, 1, 1, 1))),

              ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
              ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
              ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

LINESTYLES = []
for name, linestyle in linestyle_ref.items():
    LINESTYLES.append(linestyle)

MARKERSIZES = [8 for _ in range(100)]

def plot_graph(plot_x_axis, plot_y_axis, labels, plot_labels=None, plot_title=None, plot_xticks =
               None, path=os.getcwd(), xmax=None, ymax=None, xmin=None, ymin=None, legend_loc=1,
               file_name='no_name', alpha=1.0, linestyle=None, log_axis=False, log_base=10,
               y_axis_format=None, marker_count=None, no_marker=True):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn-paper')
    font = {'size' : 18}
    matplotlib.rc('font', **font)

    assert(len(plot_x_axis) == len(plot_y_axis))

    num_arrays = len(plot_x_axis)
    plot_xlabel = labels[0]
    plot_ylabel = labels[1]
    if(marker_count == None):
        marker_count = [1 for _ in range(len(plot_x_axis))]

    fig = plt.figure()
    fig.subplots_adjust(right=0.99, left=0.1, top=0.97, bottom=0.1)
    curr_plot = fig.add_subplot(111)
    num_plots = len(plot_x_axis)

    for idx in range(num_plots):
        curr_plot.plot(plot_x_axis[idx], plot_y_axis[idx], marker = None if(no_marker) else MARKERS[0 if(idx==0) else len(MARKERS)%idx],
                       markersize = MARKERSIZES[0 if(idx==0) else len(MARKERSIZES)%idx], 
                       c = COLORS[idx],
                       label = None if(plot_labels is None) else plot_labels[idx],
                       linewidth=3.0, alpha=alpha, markevery=marker_count[idx])
                    #    linestyle=LINESTYLES[idx])

    # Check if you have xticks
    if(plot_xticks != None):
        plt.xticks(plot_xticks[0], plot_xticks[1])
        plt.tick_params(labelsize=16)
        # Rotate the xticks if required
        # plt.xticks(rotation=50)

    if(plot_title is not None):
        plt.title(plot_title)

    if(log_axis):
        plt.yscale('log',basey=log_base)

    if(y_axis_format):
        curr_plot.yaxis.set_major_formatter(mtick.FormatStrFormatter(y_axis_format))

    # curr_plot.legend(fontsize='large', loc=2)

    # Labels for the axis
    curr_plot.set_xlabel(plot_xlabel)
    curr_plot.set_ylabel(plot_ylabel)

    # Set the limit if needed.
    if(xmax):
        curr_plot.set_xlim(xmax=xmax)
    if(ymax):
        curr_plot.set_ylim(ymax=ymax)
    if(xmin is not None):
        curr_plot.set_xlim(xmin=xmin)
    if(ymin is not None):
        curr_plot.set_ylim(ymin=ymin)
    # curr_plot.set_ylim(ymin=-1, ymax=105)
    curr_plot.legend(fontsize=18, loc=legend_loc)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', path, file_name + '.png'), dpi=300)
    plt.close()

    ########################################## #####################
    #                Resources
    ######################################### #####################
    """
    Useful Colors in Matplotlib: [darkkhaki, palevioletred, lightseagreen,
    lightcoral, darkgoldenrod, slateblue, rebeccapurple, mediumorchid, peru,
    limegreen, acqua]

    Markers in Matplotlib: https://matplotlib.org/api/markers_api.html
    [ '.',  ',', 'o', 'v', '^', '<', '>', 's', 'p', 'P', 'h', 'H']

    """
