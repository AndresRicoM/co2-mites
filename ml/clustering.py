import matplotlib
#matplotlib.use('GTK3Agg') #Uncomment when running program through SSH.
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from datetime import datetime
import pickle
import os
from load_data import *
from load_script import*
#from matplotlib.backends import _macosx



def clusteredData(clusterNum, start, end, sensorID, desiredDimensions):
    """
    unclusteredMatrix = data4clusters(file_name, desiredDimensions) #Get data from file.
    time = np.arange(unclusteredMatrix.shape[0])
    #unclusteredMatrix = np.concatenate([unclusteredMatrix, time[:, None]], axis = 1)
    """
    unclusteredMatrix = queryTermiteServer(start, end, sensorID, desiredDimensions)
    time = np.arange(unclusteredMatrix.shape[0])
    #Normalized
    #unclusteredMatrix = normalize_mat(unclusteredMatrix)

    plt.style.use('dark_background')

    #Run K means with n clusters.
    kmeans = KMeans(n_clusters=clusterNum)
    kmeans.fit(unclusteredMatrix)
    y_kmeans = kmeans.predict(unclusteredMatrix) #This is the clustered vector.

    centers = kmeans.cluster_centers_

    #Plot each variable with respect to sequence of bike trip. (Time)
    gs = gridspec.GridSpec(len(desiredDimensions),1)
    fig = plt.figure()
    plt.title(str(clusterNum) + ' Clusters', fontsize=30)
    #plt.ylabel('Normalized Fusion desiredDimensions', fontsize=25)
    #plt.xlabel('Time', fontsize=25)
    plt.xticks([])
    plt.yticks([])
    dot_size = 16

    for numVar in range(0,len(desiredDimensions)):

        ax = fig.add_subplot(gs[numVar])
        ax.scatter(time, unclusteredMatrix[:,numVar],c=y_kmeans, cmap='rainbow', s = dot_size)
        ax.set_ylabel(desiredDimensions[numVar], size =15)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()

    return unclusteredMatrix, y_kmeans

#"""
sensorId = "8360568"
s = "2020-10-18"
e = '2020-10-21'
des = ['temperature', 'humidity', 'ambientLight', 'eco2']
print(clusteredData(5, s, e, sensorId, des))
#"""
