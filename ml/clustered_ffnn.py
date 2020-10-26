# -*- coding: UTF-8 -*-

#CO2 Regular NN

#
import tensorflow as tf                                                         #Import needed libraries
from tensorflow import keras                                                    #Machine learning
import numpy as np                                                              #Array handling
import matplotlib.pyplot as plt                                                 #Plotting
import socket                                                                   #UDP Communication
import time
import re
from load_data import *
from clustering import *

sensorId = "8360568"
s = "2020-10-18"
e = '2020-10-21'
des = ['temperature', 'humidity', 'ambientLight', 'eco2']

[initial_X, initial_Y] = clusteredData(2, s, e, sensorId, des)

[initial_X, initial_Y] = shuffle_vect(initial_X, initial_Y)

(x_train, y_train, x_test, y_test) = split_xy(initial_X, initial_Y, .7, 0)

#Normalization
#x_train = normalize_mat(x_train)
#x_test = normalize_mat(x_test)

#y_train = normalize_vect(y_train)
#y_test = normalize_vect(y_test)

cluster_num = np.unique(initial_Y).shape[0]

print('Number of clusters is: ', cluster_num)

model = keras.Sequential([ #Declare a secuential Feed Forward Neural Network With Keras.

    keras.layers.Dense(500,input_dim = len(des) , activation = 'relu'), #input layer for the model. Takes input with six variables coming from terMITe. Adjust input_dim to add more sensors.
    #Hidden layers sequence. Each layer has 200 neurons with activation fucntion relu on every one of them .
    keras.layers.Dense(500, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(500, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(500, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #Output layer has the same number of neurons as clusters.
    keras.layers.Dense(cluster_num, activation=tf.nn.softmax, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

])

model.compile(optimizer='rmsprop', #Uses root mean squared error for optimization.
              loss='sparse_categorical_crossentropy', # soarse categorical cross entropy is used as loss function.
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split = 0.33, batch_size = 1024, epochs=20) #Epochs 60, 1000 Training can be done with different combinations of epochs depending on the data set used.
print(history.history.keys()) #terminal outout of accuracy results.

print(x_test.shape)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size =  20) #Evaluate model with test sets (X and Y).

predictions = model.predict(x_test) #Uses test set to predict.

print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.
model.summary()
model.get_config()
print ('Number of Training Examples Used: ' , y_train.size) #Helps get number of training examples used.
print ('Hours of Data;' , (y_train.size * 1.5) / 3600) #Calculates hours of data. Intervals of 1.5 seconds are used to obtain data.

plt.style.use('dark_background')

#plt.style.use('dark_background')
#Complete sript for plotting end results for accuracy on test and training set across different epochs.
plt.rcParams.update({'font.size': 25})
plt.figure(1)
plt.plot(history.history['accuracy'], '-') #Plot Accuracy Curve
plt.plot(history.history['val_accuracy'], ':')
plt.title('FFNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='lower right')
plt.show()
