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


id = '8360568'#'8360978' #8360568
directory = 'trained_models/cluster_models/' + id + '/2021-04-26 13:36:51.109107.txt'
#plt.style.use('dark_background') #Dark Style for Plot
dot_size = 10
# activate latex text rendering
plot_title = 'Clustering Algorithm Result Comparison'
unclusteredMatrix = np.loadtxt(directory,delimiter=',', dtype=float)

clustering = KMeans(n_clusters=5).fit(unclusteredMatrix[:,0:2])
y_kmeans = clustering.labels_

gs = gridspec.GridSpec(3, 2)
fig = plt.figure()
fig.suptitle(plot_title, size=20, weight="bold")

plot_title = 'Kmeans Clustering'
time_vector = np.arange(unclusteredMatrix.shape[0])
print(time_vector)
time_vector = (time_vector * 5) / 1440
print(time_vector)

ax = fig.add_subplot(gs[0,0])
ax.scatter(time_vector, unclusteredMatrix[:,0], c=y_kmeans, cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'CO2',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")
ax.set_title(plot_title, weight="bold")

ax = fig.add_subplot(gs[1,0])
ax.scatter(time_vector, unclusteredMatrix[:,1], c=y_kmeans, cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'Time (Days)',weight="bold")

ax = fig.add_subplot(gs[2,0])
ax.scatter(unclusteredMatrix[:,0], unclusteredMatrix[:,1], c=y_kmeans, cmap='rainbow', s = dot_size) #Scatter CO2
ax.set_ylabel(r'PIR',weight="bold")
ax.set_xlabel(r'CO2',weight="bold")

plot_title = 'Spectral Clustering'
#time_vector = np.arange(unclusteredMatrix.shape[0])

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
