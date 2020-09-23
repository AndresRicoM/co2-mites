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
#from matplotlib.backends import _macosx

def clusteredData(file_name):

    unclusteredMatrix = data4clusters(file_name) #Get data from file.

    #Normalized
    unclusteredMatrix = normalize_mat(unclusteredMatrix)

    plt.style.use('dark_background')

    #Run K means with n clusters.
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(unclusteredMatrix)
    y_kmeans = kmeans.predict(unclusteredMatrix) #This is the clustered vector.

    centers = kmeans.cluster_centers_

    time = np.arange(unclusteredMatrix.shape[0])

    #Plot each variable with respect to sequence of bike trip. (Time)
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure()
    plt.title('K = 4 Clustering', fontsize=30)
    #plt.ylabel('Normalized Fusion Variables', fontsize=25)
    #plt.xlabel('Time', fontsize=25)
    plt.xticks([])
    plt.yticks([])
    dot_size = 16

    ax = fig.add_subplot(gs[0])
    ax.scatter(time, unclusteredMatrix[:,0],c=y_kmeans, cmap='plasma', s = dot_size)
    ax.set_ylabel(r'Temperature', size =15)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1])
    ax.scatter(time, unclusteredMatrix[:,1],c=y_kmeans, cmap='plasma', s = dot_size )
    ax.set_ylabel(r'Humidity', size =15)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[2])
    ax.scatter(time, unclusteredMatrix[:,2],c=y_kmeans, cmap='plasma', s = dot_size)
    ax.set_ylabel(r'eCO2', size =15)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.show()

    return unclusteredMatrix.shape , y_kmeans.shape

print(clusteredData('S1.csv')) #SDC30.csv
