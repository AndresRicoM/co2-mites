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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, CuDNNLSTM    #add CuDNNLSTM to run in GPU
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from clustering import *
import random
#from tensorflow.contrib.rnn import *

"""sensorId = "8360568"
sensor2Id = "8362833"
s = "2020-11-1"
e = '2020-11-7'
des = ['eCO2'] #, 'temperature', 'humidity']
des2 = ['pir']
cluster_num = 3"""

def runRNN(sensorId, start, end, desiredValues, numberofclusters):

    #def clusteredData(clusterNum, start, end, sensorID, desiredDimensions, same_sensor, sensor2ID, desiredDimensions2, chosenAlgorithm)

    [initial_X, initial_Y] = clusteredData(numberofclusters, start, end, sensorId, desiredValues, 1, 'NA', 'NA', 's')

    #[initial_X, initial_Y] = shuffle_vect(initial_X, initial_Y)

    (x_train, y_train, x_test, y_test, x_val, y_val) = split_xy(initial_X, initial_Y, .7, 0)

    #Reshapes X matrix to be able to feed in to LSTM.
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    #print (x_train.shape)
    #print (x_test.shape)

    #Reshapes target classes for feeding into network.
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    #print (y_train.shape)
    #print (y_test.shape)

    #Begin sequential model.
    model = Sequential()

    #Fisrt layer of model LSTM. input shape expects input of the size of each X instance.

    model.add(LSTM(256, input_shape=(1,x_train.shape[2]), activation='relu', return_sequences=True)) #Uncomment to run on CPU
    model.add(Dropout(0.2))

    model.add(LSTM(256, activation = 'relu')) #Uncomment to run on CPU
    model.add(Dropout(0.2))

    #Feeds LSTM results into Dense layers for classification.
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(numberofclusters, activation='softmax')) #Only One output Unit

    #Declare optimizing fucntion and parameters.
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    #Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    #Fit model and store into history variable.
    history = model.fit(x_train, y_train, epochs=50,  batch_size = 64, validation_data=(x_val, y_val))

    print(history.history.keys()) #terminal outout of accuracy results.

    test_loss, test_acc = model.evaluate(x_test, y_test) #Evaluate model with test sets (X and Y).

    print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.

    f = open('current_model_info/current_accuracy.txt', 'w')
    f.write(str(test_acc))
    f.close()

    #Plot accuracy results of training and test data.
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 25})
    plt.figure(1)
    plt.plot(history.history['acc'], '-') #Plot Accuracy Curve - OSX (accuracy) - linux(acc)
    plt.plot(history.history['val_acc'], ':')
    number = str(random.random())
    file_name = str(sensorId) + '/train' + '/' + number + '.txt'
    np.savetxt('trained_models/cluster_models/' + file_name, history.history['acc'])
    file_name = str(sensorId) + '/test' + '/' + number + '.txt'
    np.savetxt('trained_models/cluster_models/' + file_name, history.history['val_acc'])
    plt.title('RNN Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Test Set'], loc='lower right')
    plt.savefig('Current_Training.png' , dpi = 1000)
    #plt.show()
    plt.close()
    #"""

    return model
