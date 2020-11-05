import urllib.request
import re
import json
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def queryTermiteServer(start, end, id, desiredVariables):

    url = "http://replace.media.mit.edu:8080/TermitesV2/getsinglesensorbydateid2020.php?sensorid="+id+"&start="+start+"-0-0-0-0&end="+end+"-20-0-0&project=universal"

    arr = requests.get(url).json()
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

def spread_pir(lead_set, flex_set, lead_times, flex_times): #Lead set will be the array with the most columns.
    #Convert timestaps to datetime
    lead_times = pd.to_datetime(lead_times)
    flex_times = pd.to_datetime(flex_times)

    new_set = np.zeros(lead_set.shape[0]) #Create new array with lead set size of rows.
    time = np.arange(lead_set.shape[0])

    #iterate trhough very value of the flex
    last_set = 0
    for ii in range(len(flex_times)):
        smallest_found = False
        time_delta = abs(lead_times[last_set] - flex_times[ii])
        while not smallest_found:
            if time_delta <= abs(lead_times[last_set] - flex_times[ii]):
                smallest_found = True
                new_set[last_set] = flex_set[ii]
                last_set = last_set + 1
    new_set = new_set.reshape(new_set.shape[0], 1)
    plt.style.use('dark_background')
    plt.plot(time, lead_set[:,1], c= "green")
    plt.scatter(time, new_set)
    plt.show()

    return new_set

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
