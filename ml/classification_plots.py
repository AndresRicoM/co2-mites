import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from matplotlib import gridspec
import os

sensorIds = ['8360978']#, '8361330', '8366124', '8362833', '8361377']

for id in sensorIds:

    file_number = 0
    directory = 'trained_models/ml_classifications/' + id + '/'

    for filename in os.listdir(directory): #Count number of files
        f = os.path.join(directory, filename)
        if filename != '.DS_Store':
            file_number = file_number + 1

    #print(file_number)
    plt.style.use('dark_background') #Dark Style for Plot
    dot_size = 10
    plot_title = "Sensor ID #" + id + ' Andorra Deployment'

    gs = gridspec.GridSpec(3, file_number)
    fig = plt.figure()
    fig.suptitle(plot_title, size=20)
    fixed_files = file_number
    file_number = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if filename != '.DS_Store':
            plot_title = 'Classifications for Week of: 2021-05-17 '
            input_data = np.genfromtxt(f, delimiter = ',', invalid_raise=False)
            print(input_data.shape[0])
            time_vector = np.arange(input_data.shape[0])
            print(time_vector.shape[0])

            ax = fig.add_subplot(gs[file_number])
            ax.scatter(time_vector, input_data[:,7], c=input_data[:,9], cmap='rainbow', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'CO2')
            ax.set_xlabel(r'Time')
            ax.set_title(plot_title)

            ax = fig.add_subplot(gs[file_number + fixed_files])
            ax.scatter(time_vector, input_data[:,8], c=input_data[:,9], cmap='rainbow', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'PIR')
            ax.set_xlabel(r'Time')

            ax = fig.add_subplot(gs[file_number + (2*fixed_files)])
            ax.scatter(input_data[:,7], input_data[:,8], c=input_data[:,9], cmap='rainbow', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'PIR')
            ax.set_xlabel(r'CO2')

            #ax = fig.add_subplot(gs[file_number + (3*fixed_files)])
            #ax.plot(time_vector, input_data[:,10]) #Plot CO2
            #ax.set_ylabel(r'PIR')
            #ax.set_xlabel(r'CO2')

            file_number = file_number + 1


    plt.show()
