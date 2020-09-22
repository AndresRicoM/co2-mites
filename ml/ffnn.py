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

(x_train, y_train, x_test, y_test) = create_data('S1.csv', .7, 1)

#Normalization
x_train = normalize_mat(x_train)
x_test = normalize_mat(x_test)

y_train = normalize_vect(y_train)
y_test = normalize_vect(y_test)

model = keras.Sequential([ #Declare a secuential Feed Forward Neural Network With Keras.

    keras.layers.Dense(500,input_dim = 3 , activation = 'relu'), #input layer for the model. Takes input with six variables coming from terMITe. Adjust input_dim to add more sensors.
    #Hidden layers sequence. Each layer has 200 neurons with activation fucntion relu on every one of them .
    keras.layers.Dense(500, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(500, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(500, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #Output layer has 5 neurons for each one of the five affective states. Output vector contains probabilities of classification.
    keras.layers.Dense(1, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

])

model.compile(optimizer='adam', #Uses root mean squared error for optimization.
              loss='mse', # soarse categorical cross entropy is used as loss function.
              metrics=['mse', 'mae'])

history = model.fit(x_train, y_train, validation_split = 0.33, batch_size = 20, epochs=100) #Epochs 60, 1000 Training can be done with different combinations of epochs depending on the data set used.
print(history.history.keys()) #terminal outout of accuracy results.

print(x_test.shape)
test_loss = model.evaluate(x_test, y_test, batch_size =  20) #Evaluate model with test sets (X and Y).

#print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.

predictions = model.predict(x_test) #Uses test set to predict.

model.summary()
model.get_config()
print ('Number of Training Examples Used: ' , y_train.size) #Helps get number of training examples used.
print ('Hours of Data;' , (y_train.size * 1.5) / 3600) #Calculates hours of data. Intervals of 1.5 seconds are used to obtain data.

plt.style.use('dark_background')

#plt.style.use('dark_background')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(y_test)
plt.plot(predictions)
#plt.plot(history.history['val_loss'])
plt.title('CO2')
plt.ylabel('Normalized Co2 ppm')
plt.xlabel('Time')
plt.legend(['Real', 'Fitted'], loc='upper left')
plt.show()


#Complete sript for plotting end results for accuracy on test and training set across different epochs.
"""
plt.rcParams.update({'font.size': 25})
plt.figure(1)
plt.plot(history.history['acc'], '-') #Plot Accuracy Curve
plt.plot(history.history['val_acc'], ':')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='lower right')
plt.show()
"""
