import h5py
import numpy as np
from scipy.stats import skewnorm
import argparse

# The 'generate_test_dataset' function will generate a (500 x 500 x 100) test dataset populated by 3 different signal 
# shapes (linear, skewed normal, sigmoid)
# There will be 186000 linear, 57600 skewed normal, and 6400 sigmoid signals using default variables. 

# these variables set the size of the test set, and the position and size of the squares within it.
# M1 boxes will contain signal b (skewed normal)
# M2 boxes will contain signal c (sigmoid)

ROWS, COLS, TIME = 500, 500, 100
M1_START = 6  # where to draw the first box
M1_BOX_SIZE = 12  # size of the box (length of one side)
M2_START = 10  # where to draw the first box
M2_BOX_SIZE = 4  # size of the box (length of one side)

SCALE_LINEAR = 0.333
SCALE_SKEWED = 0.666
SCALE_SIGMOID = 1.0

# main function
def generate_test_dataset(rows=ROWS, cols=COLS, time=TIME):
    matrix = np.empty((rows, cols, time), dtype=np.float32)
    a, b, c = generate_three_basic_signals()

    # create masks for signal b and c
    mask_b = create_mask(M1_START, M1_BOX_SIZE)
    mask_c = create_mask(M2_START, M2_BOX_SIZE)

    # fill the dataset
    matrix = populate_dataset(matrix, a, b, c, mask_b, mask_c)

    # generate random amplitude factors and apply to dataset
    amplification_matrix = generate_amplification_matrix()
    matrix = amplify_dataset(matrix, amplification_matrix)

    # generate random noise and apply to dataset
    noise_matrix = generate_noise_matrix()
    matrix = add_noise_to_dataset(matrix, noise_matrix)

    return matrix


# creates the signals
def generate_three_basic_signals():
    linear = np.linspace(0, SCALE_LINEAR, TIME)
    x = np.linspace(-5, 5, TIME)
    sigmoid = generate_sigmoid(x) * SCALE_SIGMOID
    skewed_norm = skewnorm.pdf(x, a=3.5)
    skewed_norm = (skewed_norm / np.max(skewed_norm)) * SCALE_SKEWED
    return linear, skewed_norm, sigmoid


def generate_sigmoid(x):
    return 1 / (1 + np.exp(-x))


# fills the test dataset
def populate_dataset(matrix, a, b, c, mask_b, mask_c):
    matrix[:] = a
    matrix[tuple(mask_b)] = b
    matrix[tuple(mask_c)] = c
    return matrix


# create masks for locations of signals b and c
def create_mask(start, size):
    x = np.arange(start, ROWS, int(ROWS / 20))
    x2 = [range(a, a + size) for a in x]
    mask1 = np.meshgrid(x2, x2)
    return mask1


# create and apply random noise to the dataset
def generate_noise_matrix():
    return np.random.normal(0, 0.05, (ROWS, COLS, TIME)).astype(np.float32)


def add_noise_to_dataset(matrix, random_matrix):
    matrix = matrix + random_matrix
    return matrix


# create and apply random amplification factors to the dataset
def generate_amplification_matrix():
    factors = np.linspace(0.666, 1.333, 1000, dtype=np.float32)
    amplification_matrix = np.random.choice(factors, (ROWS, COLS)).astype(np.float32)
    return amplification_matrix


def amplify_dataset(matrix, amplification_matrix):
    matrix = matrix * amplification_matrix[:, :, None]
    return matrix


if __name__ == "__main__":

    # Process input arguments
    parser = argparse.ArgumentParser(description='DNAe Coding Exercise')
    parser.add_argument('dataset_h5_fpath',
                        type=str,
                        help='Filepath to write dataset as a the HDF5file')
    args = parser.parse_args()

    print('[ Generating Dataset ]')
    np.random.seed(0)

    # Generate data
    data = generate_test_dataset().astype(np.float32)

    # Load dataset from HDF5 file into memory
    h5_file = h5py.File(args.dataset_h5_fpath, 'w')
    dset = h5_file.create_dataset('data', data=data)
    h5_file.flush()
    h5_file.close()
    print(' > Done')

