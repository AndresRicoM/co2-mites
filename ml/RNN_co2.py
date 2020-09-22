import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, CuDNNLSTM    #add CuDNNLSTM to run in GPU
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
#from tensorflow.contrib.rnn import *

(x_train, y_train, x_test, y_test) = create_data('S1.csv', .7, 1)

#Reshapes X matrix to be able to feed in to LSTM.
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print (x_train.shape)
print (x_test.shape)

#Reshapes target classes for feeding into network.
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print (y_train.shape)
print (y_test.shape)
print(x_train)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#Begin sequential model.
model = Sequential()

#Fisrt layer of model LSTM. input shape expects input of the size of each X instance.

model.add(LSTM(256, input_shape=(1,8), activation='relu', return_sequences=True)) #Uncomment to run on CPU
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

model.add(Dense(1, activation='relu')) #Only One output Unit

#Declare optimizing fucntion and parameters.
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

#Compile model
model.compile(
    loss='root_mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)

#Fit model and store into history variable.
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

print(history.history.keys()) #terminal outout of accuracy results.

test_loss, test_acc = model.evaluate(x_test, y_test) #Evaluate model with test sets (X and Y).

print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.

#Plot accuracy results of training and test data.
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 25})
plt.figure(1)
plt.plot(history.history['acc'], '-') #Plot Accuracy Curve
plt.plot(history.history['val_acc'], ':')
plt.title('Model Accuracy U6')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='lower right')
plt.show()
