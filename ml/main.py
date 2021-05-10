
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
from send_server import*

#Fix epochs in RNN / Fix daysForModel
minimumAcceptedAccuracy = .6
cluster_num = 5 #Indicate number of clusters that Spectral will use. Clusters become the categories for classification from the RNN.
daysForModel = 7 #Days - Variable sets days that have to go by for new model to be generated.
timeUpadateModel = 3 #Days - Variable sets the amount of days that are taken into account for a new model.
sensID = 8366124

def queryServer():
    sensorId = "8366124"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(minutes=5)
    startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    des = ['eCO2', 'pir']
    newSensorData, newSensorTimes  = queryTermiteServer(startString, endString, sensorId, des)
    return newSensorData

def getNewModel():
    #cluster_num = 4
    sensorId = "8366124"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(days=daysForModel)
    startString = s.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    des = ['eCO2', 'pir']
    return runRNN(sensorId, startString, endString, des, cluster_num)

def getMax():
    sensorId = "8366124"
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
    print(os.path.abspath(os.curdir))
    print(welcome_message);
    print('Initializing Training...')
    newModel = getNewModel()
    current_file_name = str(datetime.datetime.now())
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

    while True:

        current_time = datetime.datetime.now()
        #print('Current Model Accuracy is: ', currentAcc)

        if (abs(current_time - last_trained) > datetime.timedelta(days=daysForModel)) or currentAcc < minimumAcceptedAccuracy:
            print('Updating Model...')
            print('Initializing Training...')
            newModel = getNewModel()
            f = open('current_model_info/current_accuracy.txt')
            currentAcc = float(f.readline())
            f.close()
            f = open('trained_models/accuracy/' + current_file_name + '_accuracy.txt', 'a')
            f.write(str(currentAcc) + "\n")
            f.close()
            current_file_name = str(datetime.datetime.now())
            print('Established New Centroid Model.')
            haveModel = True
            needUpdate = False
            last_trained = datetime.datetime.now()
            currentMaxCo2 , currentMaxPir = getMax()
            print(currentMaxCo2)
            print(currentMaxPir)

        if haveModel:
            print('Receiving CO2...')
            receivedData = queryServer()

            try:
                serverPacket = ()
                serverPacket = serverPacket + (datetime.datetime.now(),)

                receivedData = np.reshape(receivedData[0],(1,2))

                serverPacket = serverPacket + (int(receivedData[0,0]),)
                serverPacket = serverPacket + (int(receivedData[0,1]),)

                receivedData[0,0] = np.true_divide(receivedData[0,0], currentMaxCo2)
                receivedData[0,1] = np.true_divide(receivedData[0,1], currentMaxPir)
                receivedData = np.reshape(receivedData, (receivedData.shape[0], 1, receivedData.shape[1])) #(receivedData.shape[0], 1, receivedData.shape[1])
                print('Received Current Room Status')
                print(receivedData)
                print('Making Desicion...')
                predictionVect = newModel.predict(receivedData)
                print(predictionVect)
                
                print('Building new packet')
                identifiedCluster = np.argmax(predictionVect)
                serverPacket = serverPacket + (int(identifiedCluster),)
                serverPacket = serverPacket + (float(np.amax(predictionVect)),) #Fix for conficence value. 
                serverPacket = serverPacket + (int(sensID),)


                #f = open('trained_models/predictions/' + current_file_name + '_predictions.txt', 'a')
                #f.write(str(predictionVect) + "\n")
                #f.close()
                #f = open('trained_models/values/' + current_file_name + '_received.txt', 'a')
                #f.write(str(receivedData) + "\n")
                #f.close()

                print('Uploading Data into Cluster Database:')
                send2server(serverPacket)

                print('Waiting for New Data...')
                time.sleep(60*15) #Wait for 15 minutes for New Prediction


            except:
                print('No New Data')           #print(newModel.predict(receivedData[0]))
