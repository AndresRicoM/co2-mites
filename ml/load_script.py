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
#

import urllib.request
import re
import json
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def queryTermiteServer(start, end, id, desiredVariables):

    url = "http://replace.media.mit.edu:8080/TermitesV2/getsinglesensorbydateid2020.php?sensorid="+id+"&start="+start+"-0-0-0-0&end="+end+"-0&project=universal"

    try:
        arr = requests.get(url).json()
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

    return  np.transpose(output_data), termiteTimes #np.any(np.isnan(output_data)), np.all(np.isfinite(output_data))
 #np.transpose(output_data)

#############################################################################
"""
sensorId = '8362833'
s = '2020-11-1'
e = '2020-11-3'
des = ['pir']

pir_data, pir_times = queryTermiteServer(s, e, sensorId, des)

sensorId = '8360568'
s = '2020-11-1'
e = '2020-11-3'
des = ['eCO2']


main_data, main_times = queryTermiteServer(s, e, sensorId, des)

print(main_data.shape)

function_test = spread_pir(main_data, pir_data)

print(function_test)
#"""
