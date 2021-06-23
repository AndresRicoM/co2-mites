import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from matplotlib import gridspec
import os

sensorIds = ['8360978', '8361330', '8366124', '8362833', '8361377']

for id in sensorIds:

    file_number = 0
    directory = 'trained_models/cluster_models/' + id + '/'

    for filename in os.listdir(directory): #Count number of files
        f = os.path.join(directory, filename)
        if filename != '.DS_Store':
            file_number = file_number + 1

    #print(file_number)
    plt.style.use('dark_background') #Dark Style for Plot
    dot_size = 5
    plot_title = "Sensor ID #" + id + ' Andorra Deployment'

    gs = gridspec.GridSpec(3, file_number)
    fig = plt.figure()
    fig.suptitle(plot_title, size=20)
    fixed_files = file_number
    file_number = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if filename != '.DS_Store':
            plot_title = 'Week of: ' + filename[0:11]
            input_data = np.genfromtxt(f, delimiter = ',', invalid_raise=False)
            print(input_data.shape[0])
            time_vector = np.arange(input_data.shape[0])
            print(time_vector.shape[0])

            ax = fig.add_subplot(gs[file_number])
            ax.scatter(time_vector, input_data[:,0], c=input_data[:,2], cmap='viridis', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'CO2')
            ax.set_xlabel(r'Time')
            ax.set_title(plot_title)

            ax = fig.add_subplot(gs[file_number + fixed_files])
            ax.scatter(time_vector, input_data[:,1], c=input_data[:,2], cmap='viridis', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'PIR')
            ax.set_xlabel(r'Time')

            ax = fig.add_subplot(gs[file_number + (2*fixed_files)])
            ax.scatter(input_data[:,0], input_data[:,1], c=input_data[:,2], cmap='viridis', s = dot_size) #Scatter CO2
            ax.set_ylabel(r'PIR')
            ax.set_xlabel(r'CO2')

            file_number = file_number + 1


    plt.show()


"""
input_data = np.genfromtxt(test_data_path + test_file_name, delimiter = ',', invalid_raise=False)

print('Getting Data Stats...')

print('Shape of input data is: ' , input_data.shape)
print('Size of input data is: ' , input_data.size)
#print('Total time of collection: ', (rm[rm.shape[0] - 1,0] - rm[0,0]) / 60000)
#print('Frequency of collection: ', rm.shape[0] / ((rm[rm.shape[0] - 1,0]) / 1000))

print('Generating Data Visualization')

plt.style.use('dark_background')

input_data[:,0] = input_data[:,0] / 60000
dot_size = 10

gs = gridspec.GridSpec(4,1)
fig = plt.figure()
fig.suptitle(plot_title, size=20)

ax = fig.add_subplot(gs[0])
ax.scatter(input_data[:,0], input_data[:,1], c=input_data[:,5], cmap='viridis', s = dot_size)
ax.set_ylabel(r'Temperature', size =16)

ax = fig.add_subplot(gs[1])
ax.scatter(input_data[:,0], input_data[:,2], c=input_data[:,5], cmap='viridis', s = dot_size)
ax.set_ylabel(r'Light', size =16)

ax = fig.add_subplot(gs[2])
ax.scatter(input_data[:,0], input_data[:,3], c=input_data[:,5], cmap='viridis', s = dot_size)
ax.set_ylabel(r'TVOC', size =16)

ax = fig.add_subplot(gs[3])
ax.scatter(input_data[:,0], input_data[:,4], c=input_data[:,5], cmap='viridis', s = dot_size)
ax.set_ylabel(r'eCO2', size =16)
ax.set_xlabel(r'Time', size =16)


plt.show()

"""
