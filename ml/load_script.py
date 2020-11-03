import urllib.request
import re
import json
import requests
import numpy as np
import pandas as pd

def queryTermiteServer(start, end, id, desiredVariables):

    url = "http://replace.media.mit.edu:8080/TermitesV2/getsinglesensorbydateid2020.php?sensorid="+id+"&start="+start+"-0-0-0-0&end="+end+"-20-0-0&project=universal"

    arr = requests.get(url).json()
    termiteData = arr[0]

    termiteName = termiteData['name']
    termiteValues = termiteData['values']
    termiteTimes = termiteData['time']
    termiteTimes = np.array(termiteTimes)
    #print(termiteTimes.shape)

    output_data = termiteTimes
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

    #output_data = np.delete(output_data, 0, axis = 0)

    return  np.transpose(output_data) #np.any(np.isnan(output_data)), np.all(np.isfinite(output_data))
 #np.transpose(output_data)

def structure_pir(main_data, pir_data):
    print(main_data.shape)
    print(pir_data.shape)
    return main_data

sensorId = '8362833'
s = '2020-10-20'
e = '2020-11-1'
des = ['pir']

pir_data = queryTermiteServer(s, e, sensorId, des)

sensorId = '8360568'
s = '2020-10-20'
e = '2020-11-1'
des = ['eco2']

main_data = queryTermiteServer(s, e, sensorId, des)

print(type(main_data[0,0]))
print(pir_data.shape)
