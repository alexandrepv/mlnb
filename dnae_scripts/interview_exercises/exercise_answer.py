# Core modules for exercise
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import h5py
from matplotlib import colors

# Extra modules for integration
import argparse

# =============================================================
#                        Exercise 1
# =============================================================

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16

def extract_features(raw_signals: np.ndarray) -> dict:

    """
    This function extracts features along the "signals" dimension and return them as individual 2D matrices

    :param raw_signals: Numpy ndarray (rows, cols, samples)
    :return: dictionary with all features (3x Numpy ndarray (rows, cols) <float32>)
    """

    features = {'mean': np.mean(raw_signals, axis=2).astype(np.float32),
                'max': np.max(raw_signals, axis=2).astype(np.float32),
                'std': np.std(raw_signals, axis=2).astype(np.float32)}

    return features


def visualise_features(features: dict) -> None:

    """
    This function generates a figure with each of the features in the dictionary as an subplot image.
    They keys of the dictionary are expected to be tha feature labels

    :param features: dictionary with all features ( Numpy ndarray (rows, cols) <float32>)
    :return:
    """

    fig = plt.figure(figsize=(18, 6), dpi=80)

    for i, key in enumerate(features.keys()):
        plt.subplot(1, len(features.keys()), i+1)
        plt.title(f'Feature "{key}"', fontsize=TITLE_SIZE)
        plt.imshow(features[key])

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# =============================================================
#                        Exercise 2
# =============================================================

def cluster_data(feature: np.ndarray, num_clusters=3, state=0) -> np.ndarray:

    """
    This function separates the values from the feature matrix into "num_cluster" classes

    :param feature: Numpy ndarray (rows, cols) <float32>
    :param num_clusters: int -> Number of clusters to separate the data into
    :param state: int -> initial random see to start the KMeans algorithm
    :return: Numpy array (rows * cols,) <int32>
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=state).fit(feature.reshape(-1, 1))

    return kmeans.labels_

def visualise_clustered_data(feature: np.ndarray, labels_1d: np.ndarray, feature_label:str, num_bins=25) -> None:

    """
    This function generates a figure with an image of the location of the labels, and a histogram with the
    distribution of the features values per class

    :param feature: Numpy ndarray (rows, cols) <float32>
    :param labels_1d: Numpy array (rows * cols,) <int32>
    :param feature_label: str -> label to add to the plot title
    :param num_bins: int -> Number of bins for the histogram
    :return: None
    """

    if feature.size != labels_1d.size:
        raise Exception('Input features and labels must contain the same number of elements')

    # Setup
    feature_1d = feature.flatten()
    labels_2d = labels_1d.reshape(feature.shape)
    plot_colors = ['tab:blue', 'tab:orange', 'tab:green']
    cmap = colors.ListedColormap(plot_colors)

    # Plotting
    fig = plt.figure(figsize=(16, 9), dpi=80)

    plt.subplot(1, 2, 1)
    plt.imshow(labels_2d, cmap=cmap)
    plt.title(f'Feature "{feature_label}" - Clusters', fontsize=TITLE_SIZE)

    plt.subplot(1, 2, 2)
    unique_labels = np.unique(labels_1d)
    for label, color in zip(unique_labels, plot_colors):
        selected_features = feature_1d[labels_1d == label]
        plt.hist(selected_features, bins=num_bins, color=color)

    plt.title(f'Feature "{feature_label}" - Histogram', fontsize=TITLE_SIZE)
    plt.xlabel('Feature Value', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Num. Pixels', fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# =============================================================
#                        Exercise 3
# =============================================================

def visualise_clustered_mean_signals(raw_signals: np.ndarray, labels_1d: np.ndarray, feature_label: str) -> None:

    """
    This function generates a figure with plots representing the mean signal of each class.

    :param raw_signals: Numpy ndarray (rows, cols, samples) <float32>
    :param labels_1d: Numpy array (rows * cols, ) <float32>
    :param feature_label: str -> label to add to the plot title
    :return:
    """

    if raw_signals.shape[0]*raw_signals.shape[1] != labels_1d.size:
        raise Exception('Input and labels must contain the same number of elements')

    # Setup
    num_samples = raw_signals.shape[2]
    signals = raw_signals.reshape(-1, num_samples)

    # Plotting
    fig = plt.figure(figsize=(16, 9), dpi=80)
    unique_labels = np.unique(labels_1d)
    for label in unique_labels:
        mean_signal = np.mean(signals[labels_1d == label, :], axis=0)
        plt.plot(mean_signal, linewidth=2)
    plt.title(f'Feature "{feature_label}" - Average Signals per cluster', fontsize=TITLE_SIZE)
    plt.xlabel('Samples', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Voltage', fontsize=AXIS_LABEL_SIZE)
    plt.legend(unique_labels)
    plt.xlim(0, num_samples-1)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# =============================================================
#                          Main
# =============================================================

if __name__ == "__main__":

    """
    This is meant to be used a reference solution, not the "correct" solution.
    The candidate may achieve the same results, or even better ones, using many other 
    methods and modules.
    """

    # Process input arguments
    parser = argparse.ArgumentParser(description='DNAe Coding Exercise Solution')
    parser.add_argument('dataset_h5_fpath',
                        type=str,
                        help='Filepath to the HDF5 dataset storing the data for the exercise')
    args = parser.parse_args()

    # ================ Solution starts here ===============

    h5_file = h5py.File(args.dataset_h5_fpath, 'r')
    data = h5_file['data'][:]

    features_dict = extract_features(raw_signals=data)
    visualise_features(features=features_dict)

    for key in features_dict.keys():

        cluster_labels_1d = cluster_data(feature=features_dict[key],
                                         num_clusters=3)

        visualise_clustered_data(feature=features_dict[key],
                                 labels_1d=cluster_labels_1d,
                                 feature_label=key)

        visualise_clustered_mean_signals(raw_signals=data,
                                         labels_1d=cluster_labels_1d,
                                         feature_label=key)

    # ================ Solution ends here ===============
