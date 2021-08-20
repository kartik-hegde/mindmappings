import math,os
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('ggplot')
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
font = {'size' : 32}
matplotlib.rc('font', **font)
MARKERS = ['^', '<', 's', 'o', '^', 'v', '>', 'p', 's', 'h', 'H']

SMALL_SIZE = 24
MEDIUM_SIZE = 32
BIGGER_SIZE = 40

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

def plot_graph(plot_x_axis, plot_y_axis, labels, plot_labels=None, plot_titles=None, plot_xticks =
               None, path=os.getcwd(), xmax=None, ymax=None, xmin=None, ymin=None, legend_loc=1,
               file_name='no_name', alpha=1.0, linestyle=None, log_xaxis=False, log_yaxis=False, log_base=10,
               y_axis_format=None, marker_line=None, markevery=None, marker=True):

    ## Shared Axis
    fig, axs = plt.subplots(2,4, figsize=(50, 15), sharex=True, sharey=False)

    for i, curr_plot in enumerate(axs.flat):

        for idx in range(len(plot_x_axis[0])):

            if(log_xaxis):
                curr_plot.set_xscale('log')
            if(log_yaxis):
                curr_plot.set_yscale('log')

            curr_plot.plot(plot_x_axis[i][idx], plot_y_axis[i][idx], 
                            label=plot_labels[idx],
                            linewidth=6.0,
                            marker=MARKERS[idx] if(marker) else None,
                            markevery=None if(markevery is None) else markevery,
                            markersize=16,
                            )

        # if(marker_line):
        #     for idx in range(len(plot_x_axis[0])):
        #         marker_x = plot_x_axis[i][idx][::markevery[idx]]
        #         marker_y = plot_y_axis[i][idx][::markevery[idx]]
        #         curr_plot.scatter(marker_x, marker_y,
        #                         marker=MARKERS[idx],
        #                         # s=16,
        #                         )

        # Check if you have xticks
        if(plot_xticks != None):
            plt.xticks(plot_xticks[0], plot_xticks[1])
            plt.tick_params(labelsize=16)
            # Rotate the xticks if required
            # plt.xticks(rotation=50)

        if(plot_titles is not None):
            curr_plot.set_title(plot_titles[i])

        if(log_yaxis):
            plt.yscale('log',basey=log_base)
        if(log_xaxis):
            plt.xscale('log',basex=log_base)

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
        # curr_plot.legend(fontsize=18, loc=legend_loc)
        curr_plot.grid()
        curr_plot.tick_params(axis = 'both', which = 'major')
        handles, legend_labels = curr_plot.get_legend_handles_labels()

    # set labels
    plt.setp(axs[-1, :])
    plt.setp(axs[:, 0])
    fig.text(0.1,0.5, labels[1], ha="center", va="center", rotation=90, fontsize=40 )
    fig.text(0.5,0.04, labels[0], ha="center", va="center", rotation=0, fontsize=40 )
    # plt.ylabel(labels[1])
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.2)  
    fig.legend(handles, legend_labels, loc='upper center',ncol=len(legend_labels))
    

    # plt.tight_layout()
    plt.savefig(os.path.join('figures', path, file_name + '.png'), dpi=300)
    print(os.path.join('figures', path, file_name + '.png'))
    plt.close()

if __name__ == "__main__":
    x0_data = range(1,100)
    y0_data = [math.log10(x) for x in x0_data]
    x1_data = range(1,100)
    y1_data = [math.log2(x) for x in x1_data]
    x_data = [[x0_data,x1_data],]*8
    y_data = [[y0_data,y1_data],]*8
    plot_graph(x_data,y_data, ['range','log'], ['10','2'], 'log')