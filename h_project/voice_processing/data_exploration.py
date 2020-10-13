import os
import librosa
import glob
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


class AudioFileClustering:

    """
    This function gives a quick visualisation on how certaing file compare to each other.
    This is a global visualisation method only
    """

    def __init__(self):

        pass

    def cluster_mfcc(self,
                     root_folder,
                     file_extension='.ogg',
                     fs=44100,
                     window_length=512,
                     window_step=256,
                     num_mfcc_features=13):

        """

        :param folder: Folder where all the audi files are
        :param file_extension:
        :param fs: Sampling frequency to interpolate the files
        :return:
        """

        # Locate all valid audio files in folder
        audio_fpaths = glob.glob(os.path.join(root_folder, f'*{file_extension}'))
        num_files = len(audio_fpaths)

        # Extract features for all the files
        features = np.zeros((num_files, num_mfcc_features * 3 + 1), dtype=np.float32)
        filenames = []
        print('[ Processing audio files for clustering]')
        for i, fpath in enumerate(audio_fpaths):
            _, filename = os.path.split(fpath)
            filenames.append(filename)
            print(f'> {i+1} / {len(audio_fpaths)}')
            signal_raw, _ = librosa.load(fpath, sr=fs)
            S = librosa.feature.melspectrogram(y=signal_raw,
                                               sr=fs,
                                               n_mels=num_mfcc_features,
                                               n_fft=window_length,
                                               hop_length=window_step)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_diff_std = np.std(np.diff(mfcc, axis=1), axis=1)
            features[i, :-1] = np.concatenate([mfcc_mean, mfcc_std, mfcc_diff_std])
            features[i, -1] = signal_raw.size

        # Scale features
        scaler = preprocessing.StandardScaler().fit(features)
        features_scaled = scaler.transform(features)

        # Fit tsne
        features_tsne = TSNE(n_components=2).fit_transform(features_scaled)
        #features_pca = PCA(n_components=2).fit_transform(features_scaled)

        # Show results
        fig, ax = plt.subplots()
        x = features_tsne[:, 0]
        y = features_tsne[:, 1]
        ax.scatter(x, y)
        plt.title('Audio File Differentiation')

        for i, txt in enumerate(filenames):
            ax.annotate(txt, (x[i], y[i]))
        plt.show()

    def cluster_avg_per_window(self,
                               folder_list,
                               file_extension='.ogg',
                               fs=44100,
                               window_step=256,
                               silence_threshold=0.00075):

        """
        This se
        :param root_folder:
        :param file_extension:
        :param fs:
        :param window_step:
        :return:
        """
        folder_features = []
        folder_filenames = []
        for root_folder in folder_list:

            # Locate all valid audio files in folder
            audio_fpaths = glob.glob(os.path.join(root_folder, f'*{file_extension}'))
            num_files = len(audio_fpaths)

            # Extract features for all the files
            filenames = []
            files_feature_vectors = np.ndarray((num_files, 2), dtype=np.float32)
            print('[ Processing audio files for clustering]')
            for i, fpath in enumerate(audio_fpaths):
                _, filename = os.path.split(fpath)
                filenames.append(filename)
                print(f'> {i + 1} / {len(audio_fpaths)}')
                signal_raw, _ = librosa.load(fpath, sr=fs, mono=True)
                signal_abs = np.abs(signal_raw)
                num_frames = signal_raw.size
                num_steps = num_frames // window_step + 1
                if num_frames % window_step != 0:
                    num_steps += 1
                windows_indices = np.arange(0, num_steps)*window_step
                windows_indices[-1] = num_frames

                # Calculate individual features for each window
                features = np.ndarray((windows_indices.size-1, 2), dtype=np.float32)
                zero_crossing_rates = librosa.feature.zero_crossing_rate(y=signal_raw,
                                                                         frame_length=512,
                                                                         hop_length=window_step)
                for j in range(windows_indices.size-1):
                    a = windows_indices[j]
                    b = windows_indices[j+1]
                    features[j, 0] = np.mean(signal_abs[a:b])
                    features[j, 1] = np.std(signal_abs[a:b])
                valid_windows = features[:, 0] > silence_threshold

                # Calculate features for this file
                files_feature_vectors[i, 0] = np.mean(features[valid_windows, 0])
                files_feature_vectors[i, 1] = np.std(features[valid_windows, 1])

                #plt.plot(signal_raw)
                #plt.plot(windows_indices[:-1], features[:, 0])
                #plt.plot(windows_indices[:-1], valid_windows*0.2)
                #plt.plot(windows_indices[:-1], zero_crossing_rates.transpose()*0.4)
                #plt.title(filename)
                #plt.show()

            folder_filenames.append(filenames)
            folder_features.append(files_feature_vectors)

        fig, ax = plt.subplots()
        for features, filenames in zip(folder_features, folder_filenames):
            plt.scatter(features[:, 0], features[:, 1])
            #for i, txt in enumerate(filenames):
            #    x = features[i, 0]
            #    y = features[i, 1]
            #    ax.annotate(txt, (x, y))
        plt.title('Mean and Std. of non-silent regions of all files')
        plt.xlabel('Mean')
        plt.ylabel('Std')
            #ax.text(x, y, txt, rotation=-45)
        plt.show()



