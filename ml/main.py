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
    cluster_num = 4
    sensorId = "8360568"
    e = datetime.datetime.now()
    endString = e.strftime('%Y'+'-'+'%m'+'-'+'%d'+'-'+'%H'+'-'+'%M'+'-'+'%S')
    s = e - datetime.timedelta(days=1)
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

    cluster_num = 3

    current_time = datetime.datetime.now()

    print(welcome_message);

    print('Initializing Training...')
    newModel = getNewModel()
    print('Established New Centroid Model.')
    haveModel = True
    needUpdate = False
    last_trained = datetime.datetime.now()

    while True:

        current_time = datetime.datetime.now()

        if abs(current_time - last_trained) > datetime.timedelta(minutes=50):
            print('Updating Model...')
            print('Initializing Training...')
            newModel = getNewModel()
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
                print(newModel.predict(receivedData))
                print('Making Desicion...')
                time.sleep(60*5)
            except:
                print('No New Data')           #print(newModel.predict(receivedData[0]))
