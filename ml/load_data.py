import numpy as np
import math
import os
import json
import csv

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
            print('Key exception occured: Not to worry!')
    return new_set

def create_data(file_name, train_percentage, shuffle):

    path = os.getcwd() + '/../data/test_data/' + file_name #Relative + absolute path to files.
    file = open(path, newline='')
    csv_reader = csv.reader(file)
    dataRows = int(len(list(file))) #Get length of file.
    file = open(path, newline='') #Restart file
    csv_reader = csv.reader(file)

    inputValues = ['temperature', 'humidity', 'pressure'] #Desired input variables.
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

def data4clusters(file_name):

    path = os.getcwd() + '/../data/test_data/' + file_name #Relative + absolute path to files.
    file = open(path, newline='')
    csv_reader = csv.reader(file)
    dataRows = int(len(list(file))) #Get length of file.
    file = open(path, newline='') #Restart file
    csv_reader = csv.reader(file)

    output_data = np.zeros(dataRows)

    inputValues = ['temperature', 'humidity', 'eCO2']

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

    """
    if shuffle:

        to_shuffle = np.hstack((x_data, y_data))
        print(to_shuffle.shape)
        #to_shuffle = np.transpose(to_shuffle)

        for i in range(50): #Shuffle Data 50 times
            np.random.shuffle(to_shuffle)

        main_data = np.delete(to_shuffle, -1, 1)
        target_data = to_shuffle[:, -1]

    else:
        main_data = np.transpose(main_data)
    """

    split = round(dataRows * train_percentage)

    x_training = x_data[:split][:]
    y_training = y_data[:split][:]

    x_test = x_data[split+1:][:]
    y_test = y_data[split+1:][:]

    return x_training, y_training, x_test, y_test

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

#print(data4clusters('SDC30.csv'))
