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

def create_data(file_name, train_percentage):

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

    main_data = np.transpose(main_data)
    main_data = np.delete(main_data, 0, 1)

    target_data = getnpfromvariable(file_name, outValues[0])

    split = round(dataRows * train_percentage)

    x_training = main_data[:split]
    y_training = target_data[:split]

    x_test = main_data[split+1:]
    y_test = target_data[split+1:]

    return x_training.shape, y_training.shape, x_test.shape, y_test.shape


print(create_data('S1.csv', .7))
