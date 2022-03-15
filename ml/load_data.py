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

import numpy as np
import math
import os
import json
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from load_script import*


def getnpfromvariable(file_name, variable_name):

    path = os.getcwd() + '/../data/test_data/' + file_name #Relative + absolute path to files.
    file = open(path, newline='')
    csv_reader = csv.reader(file)
    iterations = int(len(list(file))) #Get length of file.
    new_set = np.zeros(iterations)
    file = open(path, newline='') #Restart file
    csv_reader = csv.reader(file)

    for rows in range(iterations): #Iterate through every row of data.
        pHolder = next(csv_reader)
        pHolder = pHolder[2]
        pHolder = json.loads(pHolder)
        try:
            new_set[rows] = float(pHolder[variable_name]) #Save value as a float on new NP array.
        except:
            print('Key exception occured: Todo Sabroso no pasa nada! - AR')
    return new_set


def make_data_from_file(file_name, train_percentage, shuffle):

    path = os.getcwd() + '/../data/test_data/' + file_name #Relative + absolute path to files.
    file = open(path, newline='')
    csv_reader = csv.reader(file)
    dataRows = int(len(list(file))) #Get length of file.
    file = open(path, newline='') #Restart file
    csv_reader = csv.reader(file)

    inputValues = ['ambientLight', 'humidity', 'pressure'] #Desired input variables.
    outValues = ['eCO2']

    main_data = np.zeros(dataRows)

    for items in inputValues:
        main_data = np.vstack((main_data, getnpfromvariable(file_name, items)))

    main_data = main_data[1:][:]

    target_data = getnpfromvariable(file_name, outValues[0])

    if shuffle:

        to_shuffle = np.vstack((main_data, target_data))
        to_shuffle = np.transpose(to_shuffle)

        for i in range(50): #Shuffle Data 50 times
            np.random.shuffle(to_shuffle)

        main_data = np.delete(to_shuffle, -1, 1)
        target_data = to_shuffle[:, -1]

    else:
        main_data = np.transpose(main_data)

    split = round(dataRows * train_percentage)

    x_training = main_data[:split][:]
    y_training = target_data[:split][:]

    x_test = main_data[split+1:][:]
    y_test = target_data[split+1:][:]

    return x_training, y_training, x_test, y_test

def data4clusters(file_name, inputValues):

    path = os.getcwd() + '/../data/test_data/' + file_name #Relative + absolute path to files.
    file = open(path, newline='')
    csv_reader = csv.reader(file)
    dataRows = int(len(list(file))) #Get length of file.
    file = open(path, newline='') #Restart file
    csv_reader = csv.reader(file)

    output_data = np.zeros(dataRows)

    for items in inputValues:
        output_data = np.vstack((output_data, getnpfromvariable(file_name, items)))

    output_data = np.transpose(output_data)
    output_data = np.delete(output_data, 0, 1)

    return output_data

def normalize_mat(input_mat):
    for i in range(input_mat.shape[1]):
        maximum_value = np.max(input_mat[:,i])
        input_mat[:, i] = np.true_divide(input_mat[:,i], maximum_value)
    return input_mat

def normalize_vect(input_vect):
    maximum_value = np.max(input_vect)
    input_vect = np.true_divide(input_vect, maximum_value)
    return input_vect

def split_xy(x_data, y_data, train_percentage, shuffle):

    dataRows = x_data.shape[0]

    split = round(dataRows * train_percentage)
    validation = round(dataRows * .1)

    x_training = x_data[:split][:]
    y_training = y_data[:split][:]

    x_validation = x_data[split-validation+1:split][:]
    y_validation = y_data[split-validation+1:split][:]

    x_test = x_data[split+1:][:]
    y_test = y_data[split+1:][:]

    return x_training, y_training, x_test, y_test, x_validation, y_validation

def shuffle_vect(input_x, input_y):

    input_y = input_y.reshape(input_y.shape[0], 1)
    print(input_x.shape)
    print(input_y.shape)

    to_shuffle = np.append(input_x, input_y, axis = 1)
    print(to_shuffle.shape)
    for i in range(50): #Shuffle Data 50 times
        np.random.shuffle(to_shuffle)

    x_shuffle = np.delete(to_shuffle, -1, 1)
    y_shuffle = to_shuffle[:, -1]

    return x_shuffle, y_shuffle


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

    """
    plt.style.use('dark_background')
    plt.plot(time, lead_set[:,1], c= "green")
    plt.scatter(time, new_set)
    plt.show()
    #"""

    return new_set

def create_data_4regression(start, end, sensorID, desiredDimensions, same_sensor, sensor2ID, desiredDimensions2):
    if same_sensor: #Modify when we have only one sensor.
        (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)

    else:
        (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
        (pirMatrix , pirTimes) = queryTermiteServer(start, end, sensor2ID, desiredDimensions2)
        new_PIR = spread_pir(unclusteredMatrix, pirMatrix, unclustered_times, pirTimes)
        unclusteredMatrix = np.concatenate((unclusteredMatrix, ), axis=1)
        return unclusteredMatrix, new_PIR

def pir_regression_data(start, end, sensorID, desiredDimensions, same_sensor, sensor2ID, desiredDimensions2, classification):

    if classification:
        if same_sensor:
            (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
            #time = np.arange(unclusteredMatrix.shape[0])
            return unclusteredMatrix, unclustered_times

        else:
            (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
            (pirMatrix , pirTimes) = queryTermiteServer(start, end, sensor2ID, desiredDimensions2)
            new_PIR = spread_pir(unclusteredMatrix, pirMatrix, unclustered_times, pirTimes)
            unclusteredMatrix = np.concatenate((unclusteredMatrix, ), axis=1)
            return unclusteredMatrix, unclustered_times

    else:
        if same_sensor:
            (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
            #time = np.arange(unclusteredMatrix.shape[0])

        else:
            (unclusteredMatrix, unclustered_times) = queryTermiteServer(start, end, sensorID, desiredDimensions)
            (pirMatrix , pirTimes) = queryTermiteServer(start, end, sensor2ID, desiredDimensions2)
            new_PIR = spread_pir(unclusteredMatrix, pirMatrix, unclustered_times, pirTimes)
            unclusteredMatrix = np.concatenate((unclusteredMatrix, ), axis=1)
            #time = np.arange(unclusteredMatrix.shape[0])
            return unclusteredMatrix, new_PIR
