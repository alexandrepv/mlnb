import numpy as np
import scipy
import scipy.linalg as linalg
import scipy.signal
import time
import math

from numba import jit

def detrend(data: np.ndarray):

    """
    Fits linear regression to every row in the matrix.
    This is MUCH FASTER than using the scipy regression. Trust me.

    :param data: [isfets x signal] np.ndarray
    :return: final result, elapsed time
    """
    t0 = time.time()

    # Create basis function for linear fit
    num_data_points = data.shape[1]
    linear_curves = np.ndarray([2, num_data_points], dtype=np.float)
    linear_curves[0, :] = np.ones([num_data_points, ], dtype=np.float)
    linear_curves[1, :] = np.linspace(start=0.0, stop=1.0, num=num_data_points)
    beta_matrix = linalg.pinv2(linear_curves)
    w_matrix = np.matmul(data, beta_matrix )

    detrend_data = (data - np.matmul(w_matrix, linear_curves))

    t1 = time.time()

    return detrend_data, t1-t0

# Old Scipy implementation
def lowpass_filter(data: np.ndarray,
                   sampling_freq=20.,
                   cutoff_freq=0.2,
                   order=5):

    """
    ADAPTED FROM: https://www.programcreek.com/python/example/59508/scipy.signal.butter
    :param data_block: [isfets x time_data]
    :param sampling_freq:
    :param cutoff_freq:
    :param order:
    :return:
    """

    t0 = time.time()
    nyquist = sampling_freq / 2.

    b, a = scipy.signal.butter(order, cutoff_freq / nyquist)

    filtered_data = np.empty(data.shape)

    for i in np.arange(data.shape[0]):
        filtered_data[i, :] = scipy.signal.filtfilt(b,
                                                    a,
                                                    data[i, :],
                                                    method='gust')
    t1 = time.time()

    return filtered_data, t1-t0


def calculate_low_pass_filter_coefficients(sampling_rate=19.58,  # Hz
                                           quality_factor=0.70709999999999995,
                                           cutoff_frequency=0.5,  # Hz
                                           normalize=True):
    # http://www.ti.com/lit/an/slaa447/slaa447.pdf
    coefficients_a = 3 * [0.0]
    coefficients_b = 3 * [0.0]
    w0 = 2 * math.pi * cutoff_frequency / sampling_rate
    sin_w0 = math.sin(w0)
    cos_w0 = math.cos(w0)
    alpha = sin_w0 / (2 * quality_factor)

    coefficients_a[0] = 1 + alpha
    coefficients_a[1] = -2 * cos_w0
    coefficients_a[2] = 1 - alpha
    coefficients_b[0] = (1 - cos_w0) / 2
    coefficients_b[1] = 1 - cos_w0
    coefficients_b[2] = (1 - cos_w0) / 2

    # Normalize
    if normalize:
        coefficients_a[1] = coefficients_a[1] / coefficients_a[0]
        coefficients_a[2] = coefficients_a[2] / coefficients_a[0]
        coefficients_b[0] = coefficients_b[0] / coefficients_a[0]
        coefficients_b[1] = coefficients_b[1] / coefficients_a[0]
        coefficients_b[2] = coefficients_b[2] / coefficients_a[0]
        coefficients_a[0] = 1

    return np.array(coefficients_a, dtype=np.float32), np.array(coefficients_b, dtype=np.float32)


@jit(nopython=True)
def filter_multi_1d(data, a, b):

    """
    :param data: Input numpy ndarray in the format [isfets x signal_samples]
    :param a: biquad filter "a" coefficients
    :param b: biquad filter "b" coefficients
    :return:
    """

    # Initial definitions
    num_samples_to_average = 10
    num_sensors = data.shape[0]
    num_samples = data.shape[1]

    # TODO: Asserts are not supported here, but I need to make sure that there are at least
    #       N samples in order to calculate the average

    for j in range(num_sensors):

        start_value = np.mean(data[j, 0:num_samples_to_average])
        x1 = start_value
        x2 = start_value
        y1 = start_value
        y2 = start_value

        for i in range(num_samples):
            x0 = data[j, i]
            y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0

            data[j, i] = y0
    return data

@jit
def filter_1_d(data, a, b):
    """ Filter single vector. Biquad filter, only second order supported, because hardcoded.
    Not generic for other orders"""

    num_samples_to_average = 10

    start_value = np.mean(data[0:num_samples_to_average])
    x1 = start_value
    x2 = start_value
    y1 = start_value
    y2 = start_value

    length = data.shape[0]
    for i in range(length):
        x0 = data[i]
        y0 = b[0]*x0 + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2

        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0

        data[i] = y0
    return data


@jit
def label_signal_hysteresis(signal, output_signal_mask, thresh_top, thresh_bottom):

    # Initialize switch position
    dist_top = np.abs(signal[0] - thresh_top)
    dist_bottom = np.abs(signal[0] - thresh_bottom)
    if dist_top < dist_bottom:
        over_flag = True
    else:
        over_flag = False

    length = signal.size
    for i in range(length):
        if over_flag:
            if signal[i] < thresh_bottom:
                output_signal_mask[i] = False
                over_flag = False
            else:
                output_signal_mask[i] = True
        else:
            if signal[i] > thresh_top:
                output_signal_mask[i] = True
                over_flag = True
            else:
                output_signal_mask[i] = False

