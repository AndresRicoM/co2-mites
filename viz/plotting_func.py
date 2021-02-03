import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import mpld3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import datetime
import os

def plot_clusters(cluster_file):

    clusterData = np.loadtxt('../ml/trained_models/cluster_models/' + cluster_file)
    toPlot = ['CO2', 'PIR']

    time = np.arange(clusterData.shape[0])

    #plt.style.use('dark_background')

    #Plotting
    gs = gridspec.GridSpec(2,1)
    fig = plt.figure()
    plt.title(cluster_file[0:-11], size = 25)
    #plt.ylabel('Normalized Fusion desiredDimensions', fontsize=25)
    #plt.xlabel('Time', fontsize=25)
    plt.xticks([])
    plt.yticks([])
    dot_size = 16

    for numVar in range(0,2):

        ax = fig.add_subplot(gs[numVar])
        ax.scatter(time, clusterData[:,numVar],c=clusterData[:,2], cmap='rainbow', s = dot_size)
        ax.set_ylabel(toPlot[numVar], size =15)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    html_str = mpld3.fig_to_html(fig)
    Html_file= open("plot.html","a")
    Html_file.write(html_str)
    Html_file.close()
    #plt.show()
    plt.close()

    return

plot_clusters('2021-02-01 14:10:41.608482.txt')
