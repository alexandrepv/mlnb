import os
import json
import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Audio processing modules
import librosa
import soundfile as sf
from scipy import fft
import simpleaudio as sa

class GameDB():

    """
    This class createas a unified database for a single game.
    All Audio information can be stored in a single place for future references


    """

    def __init__(self, xml_filepath=''):

        self.db = dict()
        self.db['title'] = []
        self.db['release_date'] = ''
        self.db['developer'] = ''
        self.db['audio_files'] = []
        self.db['characters'] = []
        self.db['tags'] = []
        self.db['comments'] = ''

        if xml_filepath:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()

            # Get DEVELOPER (single)
            for node in root.findall('developer'):
                self.db['developer'] = node.text

            # Get RELEASE DATE (single)
            for node in root.findall('release_date'):
                d = int(node.get('day'))
                m = int(node.get('month'))
                y = int(node.get('year'))
                self.db['release_date'] = f'{datetime.date(y, m, d)}'

            # Get DEVELOPER (multiple)
            for node in root.findall('title'):
                title = node.text
                if title:
                    self.db['title'].append(title)

    def load_audacity_json(self, filepath:str) -> None:

        """
        Loads JSON file containing the follow format.
        {
            "labels_folder" : "C:\labels_folder",
            "audio_folder" : "C:\audio_folder",
            "label_pairs":
            [
                {"label_file": "file_1.txt", "audio_file": "audio_1.ogg"},
                {"label_file": "file_2.txt", "audio_file": "audio_2.ogg"},
                {"label_file": "file_3.txt", "audio_file": "audio_3.ogg"}
            ]
        }

        :param filepath: string describing the JSON filepath
        :return: NONE
        """
        assert os.path.exists(filepath), f"[ERROR] Filepath '{filepath}' does not exist or cannot be accessed"

        with open(filepath) as file:
            data = json.load(file)

        for entry in data['label_pairs']:

            # Open audio file
            audio_filepath = os.path.join(data['audio_folder'], entry['audio_file'])
            raw_signal, fs = sf.read(audio_filepath)

            # Create new audio entry
            new_audio = dict()
            new_audio['filepath'] = audio_filepath
            new_audio['num_frames'] = len(raw_signal)
            new_audio['sampling_frequency'] = fs
            new_audio['intervals'] = []

            # Load labels from audacity first
            labels_filepath = os.path.join(data['labels_folder'], entry['label_file'])
            labels_df = self.__load_audacity_labels(labels_filepath)

            # Add each individual interval to audio file
            for row in labels_df.itertuples():

                # Create new interval
                new_interval = dict()
                new_interval['frame_start'] = int(round(row[1] * fs))
                new_interval['frame_stop'] = int(round(row[2] * fs))
                labels = row[3].split()
                new_interval['tags'] = labels
                new_interval['comments'] = ''

                # And add new interval to new audio file
                new_audio['intervals'].append(new_interval)


            # Last, but not least, add new file to database
            self.db['audio_files'].append(new_audio)

    def load(self, filepath: str):

        """
        Reads a JSON dictionary containing all the database information
        :param filepath: filepath string
        :return:
        """

        assert os.path.exists(filepath), f"[ERROR] Filepath '{filepath}' does not exist or cannot be accessed"

        with open(filepath, 'r') as file:
            self.db = json.load(file)


    def save(self, filepath: str):

        with open(filepath, 'w') as file:
            json.dump(self.db, file, indent=2, separators=(',', ': '))

    def extract_audio_features(self, verbose=False):

        n_fft = 1024  # frame length
        n_mfcc = 40

        # Count how many audio files are there
        num_intervals = 0
        for audio in self.db['audio_files']:
            num_intervals += len(audio['intervals'])

        features = np.ndarray((num_intervals, n_mfcc))

        counter = 0
        for audio in self.db['audio_files']:

            # Load audio data
            raw_signal, fs = sf.read(audio['filepath'])

            for interval in audio['intervals']:

                a = interval['frame_start']
                b = interval['frame_stop']+1

                interval_signal = raw_signal[a:b]

                # Play audio
                #audio_16bit = (interval_signal*2**15).astype(np.int16)
                #sa.play_buffer(audio_16bit, 1, 2, fs)


                X = fft(interval_signal, n_fft)
                X_magnitude, X_phase = librosa.magphase(X)
                mfccs = librosa.feature.mfcc(y=interval_signal, sr=fs, n_mfcc=n_mfcc)
                X_magnitude_db = librosa.amplitude_to_db(X_magnitude)

                mfccs_1d = np.mean(mfccs, axis=1)

                features[counter, :] = mfccs_1d
                counter += 1

                if verbose:
                    plt.subplot(2, 2, 1)
                    plt.plot(interval_signal)
                    plt.xlim(0, b-a+1)
                    plt.subplot(2, 2, 2)
                    plt.imshow(mfccs, aspect='auto', cmap='nipy_spectral')
                    plt.title('MFCC')
                    plt.subplot(2, 2, 3)
                    plt.plot(mfccs_1d)
                    plt.title('MFCC Sums')
                    plt.subplot(2, 2, 4)
                    plt.plot(X_magnitude_db[0:int(n_fft / 2)])
                    plt.title('Fourier')
                    plt.xlim(0, n_fft/2)
                    plt.show()

        x = StandardScaler().fit_transform(features)

        X_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
        plt.title('t-SNE')
        plt.show()

        pca = PCA(n_components=2)
        pc = pca.fit_transform(x)
        plt.scatter(pc[:, 0], pc[:, 1])
        plt.title('PCA')
        plt.show()

        plt.imshow(features.transpose(), cmap='nipy_spectral')
        plt.show()

        g = 0

# ======================== Private functions =====================

    def __load_audacity_labels(self, filepath):

        # labels_df = pd.DataFrame(columns=['time_start', 'time_stop', 'label'])
        labels_df = pd.read_csv(filepath, sep='\t')
        labels_df.columns = ['time_start', 'time_stop', 'label']

        return labels_df


