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


print(getnpfromvariable('S1.csv', 'humidity').shape)
