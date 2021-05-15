
# 19 March 2021

# Data Analyst Test for DNAe

# Candidate: Efstathios Xanthopoulos


# Import the necessary libraries:
import sys
import h5py
import numpy as np
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib as mpl
import matplotlib.pyplot as plt


# Print the version of Python and Libraries:
print('Version of Python:', sys.version[:5])
print('Version of h5py:', h5py.__version__)
print('Version of Numpy:', np.__version__)
print('Version of Scikit-learn:', sklearn.__version__)
print('Version of Matplotlib:', mpl.__version__)


###################################
# Explore and understand the data #
###################################

# Open the HDF5 file:
f = h5py.File('F:/signals.h5', 'r')

# Check the keys of 'f':
print(list(f.keys()))

# Based on our observation, there is one dataset called `data` in the `signals.h5` file.
# Let's examine the dataset as a dataset object.

# Examine the dataset of 'f':
dset = f['data'][:]

# The shape of 'dset':
print('The shape of "dset":', dset.shape)

# the number of 'dset' dimensions:
print('The number of "dset" dimensions:', dset.ndim)

# The data type of 'dset':
print('The data type of "dset":', dset.dtype)

#############################
# Exploratory Data Analysis #
#############################

##################
# Exercise 1 - A #
##################

def calculate_features(matrix_3d):
    """
    This fuction extracts three features from each of the signals in the
    input dataset. The provided dataset consists of a 3D matrix of floats
    with dimensions 500x500x100 representing the x, y, and signal samples
    respectively.
    
    The function returns three 2D matrices of floats, each with the values
    of mean, maximum, and standard deviation of the third dimension from 
    the input 3D matrix, i.e. along the signal for each pixel.
    """
    
    # Calculate the values of mean for each pixel:
    mean_2d = np.mean(matrix_3d, axis=2)
    
    # Calculate the values of maximum for each pixel:
    max_2d = np.max(matrix_3d, axis=2)
    
    # Calculate the values of standard deviation for each pixel:
    std_2d = np.std(matrix_3d, axis=2)
    
    return mean_2d, max_2d, std_2d


# Generate the 2D matrix of mean values:
mean_matrix = calculate_features(dset)[0]

# Generate the 2D matrix of maximum values:
max_matrix = calculate_features(dset)[1]

# Generate the 2D matrix of standard deviation (std) values:
std_matrix = calculate_features(dset)[2]


##################
# Exercise 1 - B #
##################

def plot_features(mean_2d, max_2d, std_2d):
    """
    This function creates plots showing how the three matrices of mean,
    maximum, and standard deviation differ. 
    """
    
    # Keep the three matrices into a list:
    matrices_2d = [mean_2d, max_2d, std_2d]
    
    # Create the titles for each graph:
    titles = ['2D matrix of Mean', '2D matrix of Maximum', '2D matrix of Std']
    
    # Create a figure and a set of subplots:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    
    # Iterate over the three columns of subplots:
    for j in range(3):
        
        # Display data as an image, i.e. on a 2D regular raster (Heatmap):
        sub = ax[j].imshow(matrices_2d[j])
        fig.colorbar(sub, ax=ax[j], fraction=0.045)
        ax[j].set_xlabel('x')
        ax[j].set_ylabel('y')
        ax[j].set_title(titles[j])
    
    # Adjust automatically the subplots params: 
    fig.tight_layout()


# Plot the 2D matrices of mean, maximum, and standard deviations values:
plot_features(mean_matrix, max_matrix, std_matrix)


###################
# Data Clustering #
###################

##################
# Exercise 2 - A #
##################

def data_clustering(matrix_2d):
    """
    This function seperates the feature (mean, maximum, std) values
    into three classes by using K-Means clustering algorithm.
    """
    
    # Prepare the data for the K-Means clustering:
    data = matrix_2d.flatten().reshape(-1, 1)
    
    # Seperate the data into 3 classes:
    clusters = KMeans(n_clusters=3, random_state=0).fit(data)
    
    # Keep the labels:
    labels = clusters.labels_
    
    # Keep the min and max values for each class:
    
    # Initialise the 'min_value_class' and 'max_value_class' arrays:
    min_value_class = np.zeros(3)
    max_value_class = np.zeros(3)
    
    # Iterate over the three classes:
    for i in range(3):

        min_value_class[i] = data[labels==i].min()
        max_value_class[i] = data[labels==i].max()
        
    # Sort the min and max values of classes:
    min_value_class = np.sort(min_value_class)
    max_value_class = np.sort(max_value_class)
    
    # Initialise the 'pixel_label' to store the class of each pixel:
    pixel_label = np.zeros((500, 500))
    
    # Define the class of each pixel (well):
    for i in range(500):
        for j in range(500):
            
            # Keep the value:
            value = matrix_2d[i][j]
            
            # Examination for Class 0:
            if ((value >= min_value_class[0]) and (value <= max_value_class[0])):
                
                pixel_label[i][j] = 0
            
            # Examination for Class 1
            elif ((value >= min_value_class[1]) and (value <= max_value_class[1])):
                
                pixel_label[i][j] = 1
                
            # Examination for Class 2
            elif ((value >= min_value_class[2]) and (value <= max_value_class[2])):
                
                pixel_label[i][j] = 2
    
    return pixel_label


# Find the classes (labels) for each pixel using the mean values:
mean_labels = data_clustering(mean_matrix)

# Find the classes (labels) for each pixel using the maximum values:
max_labels = data_clustering(max_matrix)

# Find the classes (labels) for each pixel using the std values:
std_labels = data_clustering(std_matrix)


##################
# Exercise 2 - B #
##################

def plot_histograms(matrix_2d, labels, feature):
    """
    This function plots the classes created by using the K-Means
    clustering algorithm as an image and shows the histograms
    (distributions) of the feature values according to the class
    labelling.
    """
    
    # Define a list of colors:
    colors = ['#450559', '#218e8c', '#f5e61f']
    
    # Create a Colormap object generated from a list of colors.
    cmap = mpl.colors.ListedColormap(colors)
    
    # Define the legends for each class:
    legends = ['Class 0', 'Class 1', 'Class 2']
    
    # Create a figure and a set of subplots:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    
    # Iterate over the three classes:
    for i in range(3):
        
        # Display data as an image, i.e. on a 2D regular raster (Heatmap):
        sub = ax[0].imshow(labels)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Classes of 2D {} matrix'.format(feature))
        
        # Display Histograms:
        ax[1].hist(matrix_2d[labels==i].flatten(),
                   bins=30, color=colors[i],
                  label=legends[i])
        ax[1].set_xlabel('{} values'.format(feature))
        ax[1].set_ylabel('Count')
        ax[1].set_title('Histograms for each Class containing {} values'.format(feature))
        ax[1].legend()
    
    # Adjust the params of colorbar:
    cbar = fig.colorbar(sub, ax=ax[0], fraction=0.045)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['class 0', 'class 1', 'class 2'])
    
    # Adjust automatically the subplots params:
    fig.tight_layout()


# Plot the class labels of mean matrix (left graph), and
# a histogram of mean values of each class (right graph):
plot_histograms(mean_matrix, mean_labels, 'mean')


# Plot the class labels of maximum matrix (left graph), and
# a histogram of maximum values of each class (right graph):
plot_histograms(max_matrix, max_labels, 'maximum')


# Plot the class labels of std matrix (left graph), and
# a histogram of std values of each class (right graph):
plot_histograms(std_matrix, std_labels, 'std')


########################
# Signal Visualisation #
########################

##############
# Exercise 3 #
##############

def plot_signals(raw_matrix, labels, feature):
    """
    This function visualises the mean signal from each of the 
    three classes generated by the K-Means clustering algorithm.
    """
    
    # Define a list of colors:
    colors = ['#450559', '#218e8c', '#f5e61f']
    
    # Iterate over the three classes of feature:
    for i in range(3):
        
        # Plot mean signal:
        plt.plot(np.mean(dset[:, :][labels==i], axis=0),
                 color=colors[i], label='Class '+str(i))
    
    plt.xlabel('Time')
    plt.ylabel('Volt')
    plt.title('Mean Signal from ' + feature + ' feature')
    plt.legend()

    plt.show()
    

# Plot the three mean signals for all the pixels labelled for each
# of the three classes containing the mean values:
plot_signals(dset, mean_labels, 'mean')


# Plot the three mean signals for all the pixels labelled for each
# of the three classes containing the maximum values:
plot_signals(dset, max_labels, 'maximum')


# Plot the three mean signals for all the pixels labelled for each
# of the three classes containing the std values:
plot_signals(dset, std_labels, 'std')

