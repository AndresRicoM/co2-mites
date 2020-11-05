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
from load_data import *


sensorId = "8360568"
sensor2Id = "8362833"
s = "2020-10-28"
e = '2020-11-4'
des = ['eCO2', 'temperature', 'humidity', 'ambientLight']
des2 = ['pir']

[initial_X, initial_Y] = create_data_4regression(s, e, sensorId, des, 0, sensor2Id, des2)
#initial_Y = initial_Y + 1

(initial_X, initial_Y) = shuffle_vect(initial_X, initial_Y)

(x_train, y_train, x_test, y_test) = split_xy(initial_X, initial_Y, .7, 0)

#Normalization
"""
x_train = normalize_mat(x_train)
x_test = normalize_mat(x_test)


y_train = normalize_vect(y_train)
y_test = normalize_vect(y_test)
#"""

model_activation = 'relu' #'linear'

model = keras.Sequential([ #Declare a secuential Feed Forward Neural Network With Keras.

    keras.layers.Dense(500, input_dim = x_train.shape[1] , activation = 'relu'), #input layer for the model. Takes input with six variables coming from terMITe. Adjust input_dim to add more sensors.
    #Hidden layers sequence. Each layer has 200 neurons with activation fucntion relu on every one of them .
    keras.layers.Dense(500, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #keras.layers.Dense(500, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #keras.layers.Dense(500, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #keras.layers.Dense(500, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #keras.layers.Dense(500, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #Output layer has 5 neurons for each one of the five affective states. Output vector contains probabilities of classification.
    keras.layers.Dense(1, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

])

model.compile(optimizer='adam', #Uses root mean squared error for optimization.
              loss='mse', # soarse categorical cross entropy is used as loss function.
              metrics=['mse', 'mae'])

history = model.fit(x_train, y_train, validation_split = 0.33, batch_size = 20, epochs=50) #Epochs 60, 1000 Training can be done with different combinations of epochs depending on the data set used.
print(history.history.keys()) #terminal outout of accuracy results.

print(x_test.shape)
test_loss = model.evaluate(x_test, y_test, batch_size =  20) #Evaluate model with test sets (X and Y).

predictions = model.predict(x_test) #Uses test set to predict.

model.summary()
model.get_config()
print ('Number of Training Examples Used: ' , y_train.size) #Helps get number of training examples used.
print ('Hours of Data;' , (y_train.size * 5) / 60) #Calculates hours of data. Intervals of 1.5 seconds are used to obtain data.

plt.style.use('dark_background')
time = np.arange(y_test.shape[0])
#plt.style.use('dark_background')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(time, y_test, lw=.5)
plt.plot(time, predictions, lw=.5)
#plt.plot(time, x_test[:,0], c= 'g')
plt.title('PIR Activation Predictions')
plt.ylabel('PIR Activations')
plt.xlabel('Time')
plt.legend(['Real', 'Fitted'], loc='upper left')
plt.show()

good_predictions = 0
for rows in range(predictions.shape[0]):
    if (y_test[rows] - 3) < predictions[rows] < (y_test[rows]  + 3):
        #print('Good Prediction!')
        good_predictions = good_predictions + 1

print('Number of good predictions is: ', good_predictions)
print('Out of: ', predictions.shape[0] )
