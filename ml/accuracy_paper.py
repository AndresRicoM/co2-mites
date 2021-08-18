import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from matplotlib import gridspec
import os
import random

sensorIds = ['8360978', '8360568']   #['8360568', '8360978', '8361330', '8366124', '8362833', '8361377']
dot_size = 5
plot_title = "Training & Test Set Accuracies"

fig, (ax1,ax2) = plt.subplots(2,1)
fig.suptitle(plot_title, size=20, weight="bold")
for id in sensorIds:

    file_number = 0
    directory1 = 'trained_models/cluster_models/' + id + '/train' + '/'
    directory2 = 'trained_models/cluster_models/' + id + '/test' + '/'

    for filename in os.listdir(directory1): #Count number of files
        f = os.path.join(directory1, filename)
        if filename != '.DS_Store':
            file_number = file_number + 1

    #print(file_number)
    #plt.style.use('dark_background') #Dark Style for Plot

    fixed_files = file_number
    file_number = 0
    counter = 0

    all_data = np.zeros([50])
    for filename in os.listdir(directory1):
        f = os.path.join(directory1, filename)

        if filename != '.DS_Store':
            counter = counter + 1
            print(all_data.shape)
            input_data = np.genfromtxt(f, delimiter = ',', invalid_raise=False)
            print(input_data.shape)
            all_data = np.vstack((all_data, np.transpose(input_data)))

    #ax = fig.add_subplot(1,2)
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    plot_title = 'Training Set Accuracy'
    ax1.plot(all_data[1])
    ax1.plot(all_data[2])
    ax1.plot(all_data[3])
    ax1.plot(all_data[4]) #Scatter CO2"""
    ax1.set_ylabel(r'Accuracy')
    ax1.set_xlabel(r'Training Epoch', loc="right")
    ax1.set_title(plot_title , weight="bold")

    all_data = np.zeros([50])
    for filename in os.listdir(directory2):
        f = os.path.join(directory2, filename)

        if filename != '.DS_Store':
            counter = counter + 1
            print(all_data.shape)
            input_data = np.genfromtxt(f, delimiter = ',', invalid_raise=False)
            print(input_data.shape)
            all_data = np.vstack((all_data, np.transpose(input_data)))

    #ax = fig.add_subplot(1,2)
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    plot_title = 'Test Set Accuracy'
    ax2.plot(all_data[1])
    ax2.plot(all_data[2])
    ax2.plot(all_data[3])
    ax2.plot(all_data[4]) #Scatter CO2"""
    ax2.set_ylabel(r'Accuracy')
    ax2.set_xlabel(r'Training Epoch', loc="right")
    ax2.set_title(plot_title , weight="bold")

plt.show()
