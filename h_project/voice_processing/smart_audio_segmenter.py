# Core modules
import os
import glob
import re
import time

# Math modules
import pandas as pd
import librosa
import numpy as np

# Machine learning modules
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Debugging
import matplotlib.pyplot as plt
from . import default as default

class SmartAudioSegmenter():

    """
    This class uses XGBoost classifier the exact points you want to set your intervals.
    You can train the classifier using provided audio files and their respective audacity labels
    """

    def __init__(self):

        self.labels_ext = '.txt'
        self.audio_list = []
        self.target_fs = 44100
        self.scaler = StandardScaler()

        # XGBoost related variables
        self.model_voice_and_silence = None  # Detects voice versus silence
        self.model_sentence_interruption = None  # Detects any interruptions

    def load_training_data(self,
                           audio_folder,
                           labels_folder,
                           max_files=-1,
                           audio_ext='.ogg',
                           sampling_frequency=default.SAMPLING_FREQUENCY,
                           num_adjacent_windows=default.NUM_ADJACENT_WINDOWS,
                           num_mfcc_features=default.NUM_MFCC_FEATURES,
                           window_length=default.WINDOW_LENGTH,
                           window_step=default.WINDOW_STEP):

        """
        This loads all txt label files in audacity fo
        :param labels_folder:
        :param audio_folder:
        :param audio_ext:
        :return:
        """

        labels_folder_search_str = os.path.join(labels_folder, f'*{self.labels_ext}')
        audio_folder_search_str = os.path.join(audio_folder, f'*{audio_ext}')

        labels_fpath_list = glob.glob(labels_folder_search_str)

        if max_files == -1:
            max_files = len(labels_fpath_list)

        selected_labels_fpath_list = labels_fpath_list[:max_files]

        # Load labels first
        print('\n[ SmartAudioSegmenter: Loading audio and labels] ', flush=True)
        for i, label_fpath in enumerate(selected_labels_fpath_list):

            # Preprocessing
            _, label_filename = os.path.split(label_fpath)
            matches = re.search(r'game_([0-9]*)_(.*).txt', label_filename)
            game_id_str = matches[1]
            audio_filename = f'{matches[2]}{audio_ext}'
            audio_fpath = os.path.join(audio_folder, f'{audio_filename}')
            print(f' > {i + 1}/{len(selected_labels_fpath_list)} {audio_filename}', end='', flush=True)

            if not os.path.isfile(audio_fpath):
                print(' AUDIO NOT FOUND')
                continue

            # Load Audacity labels
            audacity_labels_df = pd.read_csv(label_fpath, delimiter='\t', names=["start_time", "stop_time", "label"])

            # Load audio
            raw_data, fs = librosa.load(audio_fpath, sr=self.target_fs)
            frames_and_windows = np.ndarray((audacity_labels_df.index.size, 4), dtype=np.int32)
            frames_and_windows[:, 0] = np.round(audacity_labels_df['start_time'].values * fs)
            frames_and_windows[:, 1] = np.round(audacity_labels_df['stop_time'].values * fs)
            frames_and_windows[:, 2] = frames_and_windows[:, 0] // window_step
            frames_and_windows[:, 3] = frames_and_windows[:, 1] // window_step

            intervals_df = pd.DataFrame(columns=['start_frame', 'stop_frame', 'start_window', 'stop_window'],
                                        data=frames_and_windows)
            intervals_df['label'] = audacity_labels_df['label'].values

            # Extract audio features
            features = self.extract_features_voice_and_silence(signal_raw=raw_data,
                                                               fs=sampling_frequency,
                                                               num_adjacent_windows=num_adjacent_windows,
                                                               num_mfcc_features=num_mfcc_features,
                                                               window_length=window_length,
                                                               window_step=window_step)
            num_windows = features.shape[0]

            # Label all windows
            labels = np.zeros((num_windows,), dtype=np.bool)
            stop_last_frame = -1
            for index, interval in intervals_df.iterrows():
                a = interval['start_frame'] // window_step
                b = interval['stop_frame'] // window_step + 1
                labels[a:b] = True

                if np.abs(a - stop_last_frame) <= 1:
                    labels[stop_last_frame] = False
                    labels[a] = False
                stop_last_frame = b

            # Put it all together and add to the list
            new_dict = {'filename': audio_filename,
                        'signal': raw_data,
                        'features': features,
                        'labels': labels,
                        'intervals': intervals_df,
                        'window_length': window_length,
                        'window_step': window_step,
                        'fs': fs,
                        'game_id_str': game_id_str}
            self.audio_list.append(new_dict)
            print(' OK')

    def train_model_voice_and_silence(self, n_estimators=32):

        """
        This function trains the model reponsable for detecting voice from silence in all loaded examples
        :return:
        """
        print('\n[ Training model: Voice and Silence]')
        feature_list = []
        label_list = []
        for audio in self.audio_list:
            feature_list.append(audio['features'])
            label_list.append(audio['labels'])

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list)

        ratio_positive = np.sum(labels) / len(labels)
        scale_pos_weight = (labels.size - np.sum(labels)) / np.sum(labels)
        print(f' > {len(labels)} Labels: {ratio_positive * 100:.2f}% positive / '
              f'{(1 - ratio_positive) * 100:.2f}% negative -> scale_pos_weight = {scale_pos_weight:.3f}')
        features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                    labels,
                                                                                    test_size=0.1,
                                                                                    random_state=123)

        self.model_voice_and_silence = xgb.XGBClassifier(objective='binary:logistic',
                                                         n_estimators=n_estimators,
                                                         seed=42,
                                                         scale_pos_weight=scale_pos_weight)
        self.model_voice_and_silence.fit(features_train, labels_train, )

        labels_predicted = self.model_voice_and_silence.predict(features_test)
        accuracy = float(np.sum(labels_predicted == labels_test)) / labels_test.size
        print(f" > Accuracy: {accuracy * 100:.3f}%", flush=True)

        print(self.model_voice_and_silence.feature_importances_)
        # plot
        plt.bar(range(len(self.model_voice_and_silence.feature_importances_)),
                self.model_voice_and_silence.feature_importances_)
        plt.show()

    def predict_model_void_and_silence(self,
                                       audio_folder,
                                       window_step=default.WINDOW_STEP,
                                       audio_extension='.ogg'):

        audio_folder_search_str = os.path.join(audio_folder, f'*{audio_extension}')
        audio_fpath_list = glob.glob(audio_folder_search_str)

        for i, audio_fpath in enumerate(audio_fpath_list):

            raw_data, fs = librosa.load(audio_fpath, sr=self.target_fs)
            features = self.extract_features_voice_and_silence(signal_raw=raw_data, fs=fs)

            predicted_labels = self.model_voice_and_silence.predict(features)
            predicted_proba = self.model_voice_and_silence.predict_proba(features)

            fig = plt.figure()

            x_scaled = np.arange(predicted_labels.size) * window_step

            _, filename = os.path.split(audio_fpath)

            plt.title(f"{filename}")
            plt.plot(raw_data)
            plt.plot(x_scaled, predicted_labels * 0.15)
            plt.plot(x_scaled, predicted_proba[:, 1] * 0.05)
            plt.tight_layout()
            plt.legend(['Signal','Predicted Labels', 'Predicted_probability'])
            plt.show()
            plt.close(fig)

    def extract_features_voice_and_silence(self,
                                           signal_raw,
                                           fs,
                                           num_adjacent_windows=default.NUM_ADJACENT_WINDOWS,
                                           num_mfcc_features=default.NUM_MFCC_FEATURES,
                                           window_length=default.WINDOW_LENGTH,
                                           window_step=default.WINDOW_STEP,
                                           smoothing_avg_window_len=default.SMOOTHING_AVG_WINDOW_LEN):
        """
        This function extracts the mfcc features for the signal and each window is concatenated with its adjacent windows
        for capturing temporal information
        :param signal_raw: numpy array (n_samples, )
        :param fs: float - Sampling frequency of the sound
        :param num_adjacent_windows: - Number of windows of either side to use
        :param mfcc_features: int
        :param window_length: int
        :param window_step: int
        :return:
        """

        S = librosa.feature.melspectrogram(y=signal_raw,
                                           sr=fs,
                                           n_mels=num_mfcc_features,
                                           n_fft=window_length,
                                           hop_length=window_step)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))

        num_samples = signal_raw.size
        num_windows = mfcc.shape[1]

        # Calculate minimum avg
        signal_avg = self.smoothMovingAvg(np.abs(signal_raw), windowLen=smoothing_avg_window_len)

        # Determine windows start and stop points
        window_min_avg = np.ndarray((num_windows, 1), np.float32)
        for i in range(num_windows-1):
            a = i * window_step
            b = np.clip((i + 1) * window_step, 0, num_samples)
            window_min_avg[i] = np.min(signal_avg[a:b])
        a = (num_windows-1) * window_step
        b = np.clip(num_windows * window_step, 0, num_samples)
        window_min_avg[-1] = np.min(signal_avg[a:b])

        #num_base_features = mfcc.shape[0] + 1
        num_base_features = 1

        if num_adjacent_windows == 0:
            features = window_min_avg
        else:
            num_extended_features = num_base_features * (1 + 2 * num_adjacent_windows)  # +1 is the avg
            features_in = window_min_avg
            features_out = np.zeros((num_windows, num_extended_features), dtype=np.float32)
            start = num_adjacent_windows
            stop = num_windows - num_adjacent_windows
            for i in range(start, stop):
                a = i - num_adjacent_windows
                b = i + num_adjacent_windows + 1
                selection_features = features_in[a:b, :]
                features_out[i, :] = selection_features.flatten()

            # Fix beginning and end of features
            for i in range(num_adjacent_windows):
                features_out[i, :] = features_out[i + 1, :]
                features_out[-(i + 1), :] = features_out[-(i + 2), :]
            features = features_out

        return features

    def smoothMovingAvg(self, inputSignal, windowLen=11):

        windowLen = int(windowLen)
        if inputSignal.ndim != 1:
            raise ValueError("")
        if inputSignal.size < windowLen:
            raise ValueError("Input vector needs to be bigger than window size.")
        if windowLen < 3:
            return inputSignal
        s = np.r_[2 * inputSignal[0] - inputSignal[windowLen - 1::-1],
                  inputSignal, 2 * inputSignal[-1] - inputSignal[-1:-windowLen:-1]]
        w = np.ones(windowLen, 'd')
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[windowLen:-windowLen + 1]

    def test_model_voice_and_silence(self):

        for i, audio in enumerate(self.audio_list):
            features, labels = self.prepare_features_voice_and_silence([i])

            predicted_labels = self.model_voice_and_silence.predict(features)
            predicted_proba = self.model_voice_and_silence.predict_proba(features)

            fig = plt.figure()

            x_scaled = np.arange(predicted_labels.size) * self.window_step

            plt.title(f"{audio['filename']}")
            plt.plot(audio['signal'])
            plt.plot(x_scaled, labels*0.2)
            plt.plot(x_scaled, predicted_labels*0.15)
            plt.plot(x_scaled, predicted_proba[:, 1]*0.05)
            plt.tight_layout()
            plt.legend(['Signal', 'Target Labels', 'Predicted Labels', 'Predicted_proba'])
            plt.show()
            plt.close(fig)

    def demo_segment_audio_file(self, audio_fpath):

        pass

    def save_model(self, fpath):

        self.model.save(fpath)




