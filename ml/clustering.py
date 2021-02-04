import matplotlib
#matplotlib.use('GTK3Agg') #Uncomment when running program through SSH.
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import datetime
import pickle
import os
from load_data import *
from load_script import*
#from matplotlib.backends import _macosx



def clusteredData(clusterNum, start, end, sensorID, desiredDimensions, same_sensor, sensor2ID, desiredDimensions2, chosenAlgorithm):
    """
    unclusteredMatrix = data4clusters(file_name, desiredDimensions) #Get data from file.
    time = np.arange(unclusteredMatrix.shape[0])
    #unclusteredMatrix = np.concatenate([unclusteredMatrix, time[:, None]], axis = 1)
    """
    #give_me_data(start, end, sensorID, desiredDimensions, same_sensor, sensor2ID, desiredDimensions2,classification):
    if same_sensor:
        (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)

    else:
        (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
        (pirMatrix , pirTimes) = queryTermiteServer(start, end, sensor2ID, desiredDimensions2)
        new_PIR = spread_pir(unclusteredMatrix, pirMatrix, unclustered_times, pirTimes)
        #print(new_PIR.shape)
        #print(unclusteredMatrix.shape)
        unclusteredMatrix = np.concatenate((unclusteredMatrix, new_PIR), axis=1)

    time = np.arange(unclusteredMatrix.shape[0])

    #Normalized
    unclusteredMatrix = normalize_mat(unclusteredMatrix)

    plt.style.use('dark_background')

    if chosenAlgorithm == 'k':
        print('KMEANS CLUSTERING')
        clustering = KMeans(n_clusters=clusterNum, random_state=0).fit(unclusteredMatrix)
        y_kmeans = clustering.labels_
        #centers = clustering.cluster_centers_

    if chosenAlgorithm == 's':
        print('SPECTRAL CLUSTERING')
        clustering = SpectralClustering(n_clusters=clusterNum, assign_labels="discretize",random_state=0).fit(unclusteredMatrix)
        y_kmeans = clustering.labels_
        file_name = str(datetime.datetime.now()) + '.txt'
        np.savetxt('trained_models/cluster_models/' + file_name,np.hstack((unclusteredMatrix, np.reshape(y_kmeans,(y_kmeans.shape[0], 1)))),delimiter=",")
        print('New Clustered Data has been saved!')
        #centers = clustering.cluster_centers_

    if chosenAlgorithm == 'db':
        print('DBSCAN')
        clustering = DBSCAN(eps=.01, min_samples=2).fit(unclusteredMatrix)
        y_kmeans = clustering.labels_
        #centers = clustering.cluster_centers_

    if chosenAlgorithm == 'a':
        print('AGGLOMERATIVE CLUSTERING')
        clustering = AgglomerativeClustering(n_clusters=clusterNum).fit(unclusteredMatrix)
        y_kmeans = clustering.labels_
        #centers = clustering.cluster_centers_

    #
    #Plotting
    if same_sensor:
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

        #plt.show()
        plt.savefig('../viz/Current_Clusters.png' , dpi = 1000)
        plt.close()

    else:
        gs = gridspec.GridSpec(len(desiredDimensions)+1,1)
        fig = plt.figure()
        plt.title(str(clusterNum) + ' Clusters', fontsize=30)
        #plt.ylabel('Normalized Fusion desiredDimensions', fontsize=25)
        #plt.xlabel('Time', fontsize=25)
        plt.xticks([])
        plt.yticks([])
        dot_size = 16

        for numVar in range(0,len(desiredDimensions)+1):

            if numVar > len(desiredDimensions)-1:
                print()
                ax = fig.add_subplot(gs[numVar])
                ax.scatter(time, unclusteredMatrix[:,numVar],c=y_kmeans, cmap='Pastel1', s = dot_size)
                ax.set_ylabel(desiredDimensions2[0], size =15)
                ax.set_yticklabels([])
                ax.set_xticklabels([])

            else:
                ax = fig.add_subplot(gs[numVar])
                ax.scatter(time, unclusteredMatrix[:,numVar],c=y_kmeans, cmap='Pastel1', s = dot_size)
                ax.set_ylabel(desiredDimensions[numVar], size =15)
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        #plt.show()
        plt.savefig('../viz/Current_Clusters.png' , dpi = 1000)
        plt.close()
        #

    return unclusteredMatrix, y_kmeans

def get_centroids(incomingMatrix, clusterNum):
    print('SPECTRAL CLUSTERING')
    clustering = SpectralClustering(n_clusters=clusterNum, assign_labels="discretize",random_state=0).fit(incomingMatrix)
    y_kmeans = clustering.labels_
    centers = clustering.cluster_centers_
    return centers

def loopqueryTermiteServer(start, end, id, desiredVariables):

    url = "http://replace.media.mit.edu:8080/TermitesV2/getsinglesensorbydateid2020.php?sensorid="+id+"&start="+start+"-0-0-0-0&end="+end+"-0&project=universal"

    try:
        arr = requests.get(url).json()
        #arr = requests.get(url)
        #print('this is the received array.    ', arr)
    except:
        print('Could Not Connect To terMITe Server! =( ')

    termiteData = arr[0]

    termiteName = termiteData['name']
    termiteValues = termiteData['values']
    termiteTimes = termiteData['time']
    termiteTimes = np.array(termiteTimes)
    #print(termiteTimes.shape)

    output_data = np.arange(len(termiteValues))
    #print(output_data.shape)

    for items in desiredVariables:
        holdingVector = np.zeros(len(termiteValues))
        for rows in range(len(termiteValues)):
            pHolder = termiteValues[rows]
            try:
                holdingVector[rows] = float(pHolder[items]) #Save value as a float on new NP array.
            except:
                print('Key exception occured: Todo Sabroso no pasa nada, besos y animo! - AR')

        output_data = np.vstack((output_data, holdingVector))

    output_data = np.delete(output_data, 0, axis = 0)
    #print('Test Data!     ', output_data)
    return  np.transpose(output_data), termiteTimes #np.any(np.isnan(output_data)), np.all(np.isfinite(output_data))
 #np.transpose(output_data)
"""
sensorId = "8360568"
sensor2Id = "8362833"
s = "2020-11-6"
e = '2020-11-7'
des = ['eCO2', 'temperature', 'humidity']
des2 = ['pir']
print(clusteredData(2, s, e, sensorId, des, 0, sensor2Id, des2 , chosenAlgorithm = 's'))
#"""
