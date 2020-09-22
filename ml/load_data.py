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
        new_set[rows] = float(pHolder[variable_name]) #Save value as a float on new NP array.

    return new_set

def tt_split(set, train_percentage): #Fubction for spliting data into training and test set
    test = 2 * set
    train = 3 * set
    return test, train

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


def normalize_mat(input_mat):
    for i in range(input_mat.shape[1]):
        maximum_value = np.max(input_mat[:,i])
        input_mat[:, i] = np.true_divide(input_mat[:,i], maximum_value)
    return input_mat

def normalize_vect(input_vect):
    maximum_value = np.max(input_vect)
    input_vect = np.true_divide(input_vect, maximum_value)
    return input_vect

#print(create_data('S1.csv', .7, 0))
