import os
import glob
import librosa
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import h_project.voice_processing.default as default
from h_project.voice_processing.audio_segmentation.audio_segmenter import AudioSegmenter

class CharacterVoiceOrganizer():


    def __init__(self):

        pass

    def calculate_folder_means_and_std(self, root_folder, audio_segmenter, output_csv_fpath='', threshold=0.00075, window_step=128):

        files_fpaths = []
        filenames = []

        # Locate all files
        for extension in default.supported_valid_extensions:
            fpath = os.path.join(root_folder, f'*{extension}')
            files_fpaths.extend(glob.glob(fpath))

        # Extract filenames
        for fpath in files_fpaths:
            _, filename = os.path.split(fpath)
            filenames.append(filename)


        all_means = np.ndarray((len(files_fpaths)), dtype=np.float32)
        all_stds = np.ndarray((len(files_fpaths)), dtype=np.float32)

        print('[ Calculating Meand and Std ]')
        for i, fpath in enumerate(files_fpaths):
            print(f'{i+1}/{len(files_fpaths)}')
            signal_raw, fs = librosa.load(fpath, sr=44100)
            frame_intervals = audio_segmenter.process(signal_raw=signal_raw,
                                                      window_size=window_step,
                                                      silence_threshold=threshold)
            means, stds = audio_segmenter.get_mean_std_from_intervals(signal_raw=signal_raw,
                                                                      frame_intervals=frame_intervals)
            all_means[i] = np.mean(means)
            all_stds[i] = np.mean(stds)

        # Write outputs
        files_df = pd.DataFrame(columns=['file', 'mean', 'std'])
        files_df['file'] = filenames
        files_df['mean'] = all_means
        files_df['std'] = all_stds

        if len(output_csv_fpath) == 0:
            output_csv_fpath = os.path.join(root_folder, 'means_and_stds.csv')
        files_df.to_csv(output_csv_fpath, index_label=False, index=False)

    def test_different_thresholds(self, root_folder, audio_segmenter, fs=44100):

        files_fpaths = []
        files_intervals = []
        files_num_samples = []

        # Control Panel
        num_thresholds = 9
        thresholds = np.arange(0, num_thresholds, 1) * 0.00015 + 0.00010
        thresholds_heights = np.linspace(0, 0.1, num_thresholds)
        window_step = 128
        legend = ['signal_raw']

        # Locate all files
        for extension in default.supported_valid_extensions:
            fpath = os.path.join(root_folder, f'*{extension}')
            files_fpaths.extend(glob.glob(fpath))

        # Segment files
        for i, fpath in enumerate(files_fpaths):
            print(f'{i+1}/{len(files_fpaths)}')
            _, filename = os.path.split(fpath)

            signal_raw, fs = librosa.load(fpath, sr=fs)
            plt.plot(signal_raw)
            for threshold, height in zip(thresholds, thresholds_heights):
                legend.append(f'{threshold:.5f}')
                intervals = audio_segmenter.process(signal_raw=signal_raw,
                                                    window_size=window_step,
                                                    silence_threshold=threshold,
                                                    debug=False)
                signal_valid = np.zeros_like(signal_raw)
                for interval in intervals:
                    signal_valid[interval[0]:interval[1]] = 1.0
                signal_valid *= height
                plt.plot(signal_valid)
            plt.legend(legend)
            plt.title(filename)
            plt.show()


        """
        # Save all data in to H5 file
        h5_file = h5py.File(output_h5_fpath, 'w')
        signal_data = h5_file.create_dataset('signal', (total_sum,), dtype=np.float32)
        all_signal_data = np.concatenate(files_intervals)
        signal_data[:] = all_signal_data
        h5_file.flush()
        h5_file.close()
        """
