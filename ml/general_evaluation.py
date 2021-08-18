import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from matplotlib import gridspec
import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import rc,rcParams

#id = '8360978'#'8360978' #8360568
directory = 'trained_models/cluster_models/8360978/2021-05-24 10/49/37.362008.txt'
dot_size = 10

plot_title = 'Live Classification and Clustering Result Comparison'
unclusteredMatrix = np.loadtxt(directory,delimiter=',', dtype=float)

gs = gridspec.GridSpec(3, 2)
fig = plt.figure()
fig.suptitle(plot_title, size=20, weight="bold")

plot_title = 'Spectral Clustering Results'
time_vector = np.arange(unclusteredMatrix.shape[0])
time_vector = (time_vector * 5) / 1440

ax = fig.add_subplot(gs[0,0])
ax.scatter(time_vector, unclusteredMatrix[:,0], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'CO2',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")
ax.set_title(plot_title, weight="bold")

ax = fig.add_subplot(gs[1,0])
ax.scatter(time_vector, unclusteredMatrix[:,1], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")

ax = fig.add_subplot(gs[2,0])
ax.scatter(unclusteredMatrix[:,0], unclusteredMatrix[:,1], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'CO2',weight="bold")

plot_title = 'Live Model Classification Results'
id = '8360978'#'8360978' #8360568
directory = 'trained_models/ml_classifications/' + id + '/2021-05-17.txt'
time_vector = np.arange(unclusteredMatrix.shape[0])
time_vector = (time_vector * 5) / 1440

ax = fig.add_subplot(gs[0,1])
ax.scatter(time_vector, unclusteredMatrix[:,0], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'CO2',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")
ax.set_title(plot_title, weight="bold")

ax = fig.add_subplot(gs[1,1])
ax.scatter(time_vector, unclusteredMatrix[:,1], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")

ax = fig.add_subplot(gs[2,1])
ax.scatter(unclusteredMatrix[:,0], unclusteredMatrix[:,1], c=unclusteredMatrix[:,2], cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'CO2',weight="bold")

plt.show()
