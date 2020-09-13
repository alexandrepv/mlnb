import argparse
import xml.etree.ElementTree as ET
import os
import time
import datetime
import glob

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt



input_path = r'C:\game_data\game_0704\audio\voice'
output_path = r'C:\game_data\game_0704\audio'

def create_game_dictionary():

    game = {'title': [],  # List of strings with multiple variables of the same time
            'release_date': '',
            'developer': '',
            'voice_files': [],  # List of dictionaries describing each audio file
            'characters': [],
            'tags': [],  # list of tags (strings)
            'comments': '',
            }

    return game

def create_voice_file_dictionary():

    voice_file = {'filepath': '',
                  'sampling_rate': 0,
                  'bitrate': 0,
                  'num_samples': 0,
                  'intervals': [],
                  'voice_actor': [],
                  'character': [],
                  'tags': [],  # List of string describing the sound, or emotion in this case
                  'comments': '',
                  'transcript': [],  # List of transcripts in english, japanese, romanji, etc
                  }

    return voice_file


def process_voice_files(filepath_list, xml_info_filepath, output_directory):

    """
    This function reads the XML description file that I created and
    :param filepath_list:
    :param xml_info_file:
    :param output_directory:
    :return:
    """

    pass

def find_intervals(valid_windows):


    """
    This function splits the valid windows into intervals,
    but first if fixes the the 'last-sample-detached' problem
    :param valid_windows: array of booleans indicating which windows are valid
    :return:
    """

    num_windows = len(valid_windows)
    new_interval = True
    current_interval_start = 0
    intervals = []

    for i in range(0, len(valid_windows)):

        # Start a new interval
        if new_interval and valid_windows[i]:
            new_interval = False
            current_interval_start = i

        # Append to current interval
        if not new_interval and not valid_windows[i]:
            current_interval_stop = i
            new_interval = True
            a = current_interval_start - 1
            if a < 0:
                a = 0
            b = current_interval_stop + 1
            if b >= num_windows:
                b = num_windows - 1

            intervals.append([a, b])

    return intervals

def split_audio_into_intervals(raw_signal, fs):

    """

    :param raw_signal:
    :param fs:
    :param threshold:
    :return: list of dictionaries
    """

    # Define constants
    windows_size = 256
    spectogram_scale = 750      # TODO: Make these two numbers one!
    spectogram_thres = 0.0012

    num_samples = len(raw_signal)

    # SPECTOGRAM
    freqs, times, spectrogram = signal.spectrogram(raw_signal, fs=fs, window='hanning',
                                          nperseg=windows_size,
                                          detrend=False, scaling='spectrum')
    spectogram_amplitudes = np.sum(spectrogram*spectogram_scale, axis=0)
    spectogram_x_axis = np.linspace(0, num_samples, len(spectogram_amplitudes))
    spectogram_valid_windows = spectogram_amplitudes >= spectogram_thres
    frames = np.round(times*fs)
    frames = frames.astype(int)

    # Step 1) Separate window intervals
    intervals = find_intervals(spectogram_valid_windows)

    offset_frames = windows_size / 2
    intervals_array = np.array(intervals)
    intervals_frames = np.zeros_like(intervals_array)
    intervals_frames[:, 0] = frames[intervals_array[:, 0]] - offset_frames
    intervals_frames[:, 1] = frames[intervals_array[:, 1]] - offset_frames

    intervals_visual = np.zeros((num_samples,))
    for i, inter in enumerate(intervals_frames):
        intervals_visual[inter[0]:inter[1]] = (i+1)*0.1

    plt.plot(raw_signal)
    plt.plot(intervals_visual*0.075)
    plt.show()

    return intervals_frames


