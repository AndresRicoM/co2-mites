# -*- coding: UTF-8 -*-
#
#    ██████╗ ██████╗ ██████╗       ███╗   ███╗██╗████████╗███████╗███████╗
#   ██╔════╝██╔═══██╗╚════██╗      ████╗ ████║██║╚══██╔══╝██╔════╝██╔════╝
#   ██║     ██║   ██║ █████╔╝█████╗██╔████╔██║██║   ██║   █████╗  ███████╗
#   ██║     ██║   ██║██╔═══╝ ╚════╝██║╚██╔╝██║██║   ██║   ██╔══╝  ╚════██║
#   ╚██████╗╚██████╔╝███████╗      ██║ ╚═╝ ██║██║   ██║   ███████╗███████║
#    ╚═════╝ ╚═════╝ ╚══════╝      ╚═╝     ╚═╝╚═╝   ╚═╝   ╚══════╝╚══════╝
#
#   ╔═╗┬┌┬┐┬ ┬  ╔═╗┌─┐┬┌─┐┌┐┌┌─┐┌─┐       ╔╦╗╦╔╦╗  ╔╦╗┌─┐┌┬┐┬┌─┐  ╦  ┌─┐┌┐
#   ║  │ │ └┬┘  ╚═╗│  │├┤ ││││  ├┤   ───  ║║║║ ║   ║║║├┤  │││├─┤  ║  ├─┤├┴┐
#   ╚═╝┴ ┴  ┴   ╚═╝└─┘┴└─┘┘└┘└─┘└─┘       ╩ ╩╩ ╩   ╩ ╩└─┘─┴┘┴┴ ┴  ╩═╝┴ ┴└─┘
#
#                                   .|
#                                  | |
#                                  |'|            ._____
#                          ___    |  |            |.   |' .---"|
#                  _    .-'   '-. |  |     .--'|  ||   | _|    |
#               .-'|  _.|  |    ||   '-__  |   |  |    ||      |
#               |' | |.    |    ||       | |   |  |    ||      |
#            ___|  '-'     '    ""       '-'   '-.'    '`      |____
#
#

from RNN_co2 import*
import datetime
import time
import os
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import StandardScaler
from load_script import*
from load_data import*

#Fix epochs in RNN / Fix daysForModel

cluster_num = 4 #Indicate number of clusters that Spectral will use. Clusters become the categories for classification from the RNN.
daysForModel = 7 #Days - Variable sets days that have to go by for new model to be generated.
timeUpadateModel = 7 #Days - Variable sets the amount of days that are taken into account for a new model.

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


if __name__ == "__main__":

    welcome_message = """

        ██████╗ ██████╗ ██████╗       ███╗   ███╗██╗████████╗███████╗███████╗
       ██╔════╝██╔═══██╗╚════██╗      ████╗ ████║██║╚══██╔══╝██╔════╝██╔════╝
       ██║     ██║   ██║ █████╔╝█████╗██╔████╔██║██║   ██║   █████╗  ███████╗
       ██║     ██║   ██║██╔═══╝ ╚════╝██║╚██╔╝██║██║   ██║   ██╔══╝  ╚════██║
       ╚██████╗╚██████╔╝███████╗      ██║ ╚═╝ ██║██║   ██║   ███████╗███████║
        ╚═════╝ ╚═════╝ ╚══════╝      ╚═╝     ╚═╝╚═╝   ╚═╝   ╚══════╝╚══════╝

       ╔═╗┬┌┬┐┬ ┬  ╔═╗┌─┐┬┌─┐┌┐┌┌─┐┌─┐       ╔╦╗╦╔╦╗  ╔╦╗┌─┐┌┬┐┬┌─┐  ╦  ┌─┐┌┐
       ║  │ │ └┬┘  ╚═╗│  │├┤ ││││  ├┤   ───  ║║║║ ║   ║║║├┤  │││├─┤  ║  ├─┤├┴┐
       ╚═╝┴ ┴  ┴   ╚═╝└─┘┴└─┘┘└┘└─┘└─┘       ╩ ╩╩ ╩   ╩ ╩└─┘─┴┘┴┴ ┴  ╩═╝┴ ┴└─┘

                                       .|
                                      | |
                                      |'|            ._____
                              ___    |  |            |.   |' .---"|
                      _    .-'   '-. |  |     .--'|  ||   | _|    |
                   .-'|  _.|  |    ||   '-__  |   |  |    ||      |
                   |' | |.    |    ||       | |   |  |    ||      |
                ___|  '-'     '    ""       '-'   '-.'    '`      |____



    """

    current_time = datetime.datetime.now()


    print(welcome_message);

    print('Initializing Training...')
    newModel = getNewModel()
    f = open('current_model_info/current_accuracy.txt')
    currentAcc = float(f.readline())
    f.close()
    print('Established New Centroid Model.')
    haveModel = True
    needUpdate = False
    last_trained = datetime.datetime.now()

    while True:

        current_time = datetime.datetime.now()
        #print('Current Model Accuracy is: ', currentAcc)

        if (abs(current_time - last_trained) > datetime.timedelta(days=daysForModel)) or currentAcc < .8:
            print('Updating Model...')
            print('Initializing Training...')
            newModel = getNewModel()
            f = open('current_model_info/current_accuracy.txt')
            currentAcc = float(f.readline())
            f.close()
            current_file_name = str(datetime.datetime.now()) + '.txt'
            print('Established New Centroid Model.')
            haveModel = True
            needUpdate = False
            last_trained = datetime.datetime.now()

        if haveModel:
            print('Receiving CO2...')
            receivedData = queryServer()
            try:
                receivedData = np.reshape(receivedData[0],(1,2))
                receivedData = np.reshape(receivedData, (receivedData.shape[0], 1, receivedData.shape[1])) #(receivedData.shape[0], 1, receivedData.shape[1])
                print('Received Current Room Status')
                print(receivedData)
                print('Making Desicion...')
                predictionVect = newModel.predict(receivedData)
                print(predictionVect)
                f = open('current_model_info/prediction_list.txt', 'a')
                f.write(str(predictionVect) + "\n")
                f.close()
                f = open('current_model_info/value_list.txt', 'a')
                f.write(str(receivedData) + "\n")
                f.close()
                print('Waiting for New Data...')
                time.sleep(60*15) #Wait for 15 minutes for New Prediction
            except:
                print('No New Data')           #print(newModel.predict(receivedData[0]))
