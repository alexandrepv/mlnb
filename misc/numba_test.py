import numpy as np
# import matplotlib.pyplot as plt

from numba import jit

@jit
def calculate_intensity_for_single_channel(
        isfets_data,
        reference_channel_index,
        isfet_channel_index,
        b,
        a,
        flow_delay_samples):
    """
    Calculates intensity for single ISFET channel

    :param isfets_data: All ISFET data (for multiple channels)
    :param reference_channel_index: Index of channel (in isfet_data) with reference channel
    :param isfet_channel_index: Index of channel (in isfet_data) with channel to analyse
    :param b: polynomial filter parameters, vector B
    :param a: polynomial filter parameters, vector A
    :param flow_delay_samples: how many samples should be dropped before offset removal and max search
    :return: single intensity value
    """
    if flow_delay_samples >= isfets_data.shape[0]:
        return np.nan

    if reference_channel_index != -1:
        isfet_channel_minus_reference = isfets_data[:, isfet_channel_index] - isfets_data[:, reference_channel_index]
    else:
        isfet_channel_minus_reference = isfets_data[:, isfet_channel_index]

    filter_1_d(b=b,
               a=a,
               x=isfet_channel_minus_reference)
    filtered_isfet_channel_minus_reference = isfet_channel_minus_reference

    baseline_offset = filtered_isfet_channel_minus_reference[flow_delay_samples]
    max_val = np.amax(filtered_isfet_channel_minus_reference[flow_delay_samples:])
    return max_val - baseline_offset


@jit
def filter_1_d(b, a, x):
    """ Filter single vector. Biquad filter, only second order supported, because hardcoded.
    Not generic for other orders"""
    x1 = x[0]
    x2 = x[0]
    y1 = x[0]
    y2 = x[0]

    length = x.shape[0]
    for i in range(length):
        x0 = x[i]
        y0 = b[0]*x0 + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2

        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0

        x[i] = y0
    return x


# ======================== [Main] =========================
a0 = 0.2513790015131591
a1 = 0.5027580030263182
a2 = 0.2513790015131591
b1 = -0.17124071441396285
b2 = 0.1767567204665992