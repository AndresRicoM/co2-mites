from RNN_co2 import*
import datetime
import time
import os
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import StandardScaler
from load_script import*
from load_data import*
from send_server import*

#Fix epochs in RNN / Fix daysForModel
minimumAcceptedAccuracy = .6
cluster_num = 5 #Indicate number of clusters that Spectral will use. Clusters become the categories for classification from the RNN.
daysForModel = 7 #Days - Variable sets days that have to go by for new model to be generated.
timeUpadateModel = 3 #Days - Variable sets the amount of days that are taken into account for a new model.
sensID = 8360568

date = datetime.datetime(2021,1,11)
change = datetime.timedelta(days=7)
date_list = [date, date + change, date + (2*change)]#2021-05-

def queryServer():
    sensorId = "8360568"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(minutes=5)
    startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    des = ['eCO2', 'pir']
    newSensorData, newSensorTimes  = queryTermiteServer(startString, endString, sensorId, des)
    return newSensorData

def getNewModel():
    #cluster_num = 4
    sensorId = "8360568"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(days=daysForModel)
    startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    des = ['eCO2', 'pir']
    return runRNN(sensorId, startString, endString, des, cluster_num)

def getMax():
    sensorId = "8360568"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(days=daysForModel)
    startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    des = ['eCO2', 'pir']
    newSensorData, newSensorTimes  = queryTermiteServer(startString, endString, sensorId, des)
    max_pir = np.max(newSensorData[:,1])
    max_co2 = np.max(newSensorData[:,0])
    return max_co2, max_pir


if __name__ == "__main__":

    print('Initializing Training...')

    counter = 0
    for i in range(0,10):
        counter = counter + 1
        print(counter)

        sensorId = "8360568"
        e = date + (counter * change)
        print(e)
        endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
        s = e - change
        print(s)
        startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
        des = ['eCO2', 'pir']
        runRNN(sensorId, startString, endString, des, cluster_num)

"""
        f = open('current_model_info/current_accuracy.txt')
        currentAcc = float(f.readline())
        f.close()
        f = open('trained_models/accuracy/' + current_file_name + '_accuracy.txt', 'a')
        f.write(str(currentAcc) + "\n")
        f.close()
        print('Established New Centroid Model.')
        haveModel = True
        needUpdate = False
        last_trained = datetime.datetime.now()
        currentMaxCo2 , currentMaxPir = getMax()
        print(currentMaxCo2)
        print(currentMaxPir)

        """
