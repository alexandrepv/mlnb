import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt

# Signal processing modules
from scipy.signal import hilbert, chirp
from scipy import signal
import soundfile as sf
import librosa
from . import dsp
from . import default

class AudioSegmenter():

    """
    This class is a mini toolbox designed specifically to segment
    """

    def __init__(self, fs=default.SAMPLING_FREQUENCY, window_size=default.WINDOW_SIZE):

        self.fs = fs  # Sampling frequency to work with
        self.window_size = window_size  # Samples for internal window computations

    def process_with_librosa(self, raw_signal):

        results = librosa.effects.split(y=raw_signal, top_db=15)

        labeled_samples = np.zeros((raw_signal.size,), dtype=np.float32)
        for interval in results:
            labeled_samples[interval[0]:interval[1]] = 0.1

        plt.figure()
        plt.plot(raw_signal)
        plt.plot(labeled_samples)
        plt.legend(['Raw Signal', 'Labels'])
        plt.show()

    def process_with_avg(self,
                         raw_signal,
                         threshold_top=default.THRESHOLD_TOP,
                         threshold_bottom=default.THRESHOLD_BOTTOM,
                         window_length=default.WINDOW_LENGTH):

        avg_signal = self.smoothMovingAvg(np.abs(raw_signal), windowLen=window_length)
        signal_mask = np.zeros_like(avg_signal, dtype=np.bool)
        dsp.label_signal_hysteresis(signal=avg_signal,
                                    output_signal_mask=signal_mask,
                                    thresh_bottom=threshold_bottom,
                                    thresh_top=threshold_top)

        valid_intervals = self.find_valid_intervals(signal_mask=signal_mask,
                                                    min_sound_interval_size=600,
                                                    min_silence_interval_size=400,
                                                    start_offset=0,
                                                    stop_offset=0)

        valid_intervals_plot = np.zeros((len(avg_signal),), dtype=np.float32)
        for interval in valid_intervals:
            print(interval)
            valid_intervals_plot[interval[0]:interval[1]] = 0.075

        plt.figure()
        plt.plot(raw_signal)
        plt.plot(avg_signal)
        plt.plot(signal_mask*0.1)
        plt.plot(valid_intervals_plot)
        plt.legend(['Raw Signal', 'avg_signal', 'Signal Mask', 'Valid Intervals'])
        plt.show()

    def find_valid_intervals(self,
                             signal_mask,
                             min_sound_interval_size,
                             min_silence_interval_size,
                             start_offset,
                             stop_offset):

        """


        :param signal_mask: Numpy bool array. True = Sound, False = Silence
        :param min_sound_interval_size:
        :param min_silence_interval_size:
        :param start_offset:
        :param stop_offset:
        :return:
        """

        num_samples = len(signal_mask)
        valid_intervals = []
        silence_intervals = []

        sound_indices = np.where(signal_mask)[0]
        sound_indices_diff = np.diff(sound_indices)
        brake_indices = np.where(sound_indices_diff > min_silence_interval_size)[0]
        index_counter = 0
        intervals = []
        start = sound_indices[0]
        for break_index in brake_indices:
            stop = sound_indices[break_index]
            intervals.append([start, stop])
            start = stop + sound_indices_diff[break_index]

        return intervals

        # Stage 1) Separate labels as intervals
        new_interval = True
        for i in range(0, len(signal_mask)):

            # Start a new interval
            if new_interval and signal_mask[i]:
                new_interval = False
                current_interval_start = i

            # Append to current interval
            if not new_interval and not signal_mask[i]:
                current_interval_stop = i
                new_interval = True
                a = current_interval_start - 1
                if a < 0:
                    a = 0
                b = current_interval_stop + 1
                if b >= num_windows:
                    b = num_windows - 1

                sound_intervals.append([a, b])

        intervals_frames = []
        for i, interval in enumerate(intervals):
            a = window_frames[interval[0]]
            b = window_frames[interval[1]]
            intervals_frames.append([a + first_frame_offset, b + last_frame_offset])

        # Stage 2) Remove invalid intervals

        return valid_intervals

    def process_with_spectogram(self, raw_signal, threshold=default.SPECTOGRAM_THRESH):

        freqs, times, spectrogram = signal.spectrogram(raw_signal,
                                                       fs=self.fs,
                                                       window='hanning',
                                                       nperseg=self.window_size,
                                                       detrend=False,
                                                       noverlap=self.window_size / 4,  # quarter window of overlap
                                                       scaling='spectrum')
        spectrogram_normalised = spectrogram / np.mean(spectrogram)

        spectogram_window_amplitudes = np.sum(spectrogram_normalised, axis=0)
        spectogram_window_frames = np.linspace(0, raw_signal.size, len(spectogram_window_amplitudes)).astype(int)

        window_frame_intervals = self.find_intervals(window_amplitudes=spectogram_window_amplitudes,
                                                     window_frames=spectogram_window_frames,
                                                     threshold=threshold)

        labeled_samples = np.zeros((raw_signal.size,), dtype=np.float32)
        for interval in window_frame_intervals:
            labeled_samples[interval[0]:interval[1]] = 0.1


        plt.figure()
        plt.plot(raw_signal)
        plt.plot(labeled_samples)
        plt.legend(['Raw Signal', 'Labels', 'Smoothed Signal', 'lpf signal'])
        plt.show()



    def find_intervals(self,
                       window_amplitudes,
                       window_frames,
                       threshold,
                       first_frame_offset=0,
                       last_frame_offset=0):

        """
        This function splits a list of amplitude values (which can be from time windows) into intervals
        of values greater or equal to the threshold. At the end, the intervals are returned based on the
        provided frames.

        :param window_amplitudes: values indicated an "amplitude for that window"
        :param window_frames: Location of each window in frames. This is assumed to be at the middle of the indow
        :param threshold: Any window amplitude values above this threshold are considered as valid
        :param first_frame_offset: Value added to the frame of the first window of a valid interval
        :param last_frame_offset: Value added to the frame of the last window of a valid interval
        :param min_allowed_gap: Min number of invalid windows to be ignored when in between two valid intervals
        :return:
        """

        # Mark all the samples that are not "silence"
        window_mask = window_amplitudes >= threshold

        num_windows = len(window_mask)
        new_interval = True
        current_interval_start = 0
        intervals = []

        # Remove any 1-window gaps in the windows
        """if num_windows > default.GAP_KERNEL_SIZE:
            kernel = np.zeros((default.GAP_KERNEL_SIZE, ), dtype=np.bool)
            kernel[0], kernel[-1] = True, True
            for i in range(1, num_windows - (1 + default.GAP_KERNEL_SIZE)):
                if np.array_equal(window_mask[i:i+default.GAP_KERNEL_SIZE], kernel):
                    window_mask[i:i+default.GAP_KERNEL_SIZE] = True

        # Remove any windows of lengh 1
        if num_windows > default.GAP_KERNEL_SIZE:
            kernel = np.zeros((default.GAP_KERNEL_SIZE,), dtype=np.bool)
            kernel[1] = True
            for i in range(1, num_windows - (1 + default.GAP_KERNEL_SIZE)):
                if np.array_equal(window_mask[i:i + default.GAP_KERNEL_SIZE], kernel):
                    window_mask[i:i + default.GAP_KERNEL_SIZE] = False"""

        for i in range(0, len(window_mask)):

            # Start a new interval
            if new_interval and window_mask[i]:
                new_interval = False
                current_interval_start = i

            # Append to current interval
            if not new_interval and not window_mask[i]:
                current_interval_stop = i
                new_interval = True
                a = current_interval_start - 1
                if a < 0:
                    a = 0
                b = current_interval_stop + 1
                if b >= num_windows:
                    b = num_windows - 1

                intervals.append([a, b])

        intervals_frames = []
        for i, interval in enumerate(intervals):
            a = window_frames[interval[0]]
            b = window_frames[interval[1]]
            intervals_frames.append([a+first_frame_offset, b+last_frame_offset])

        return intervals_frames

    def extract_mfcc_features(self, raw_signal):

        """
        Extract mean, std and mean derivative of mfcc features for the duration of the signal
        :param raw_signal:
        :return:
        """

        mfccs = librosa.feature.mfcc(y=raw_signal,
                                     sr=self.fs,
                                     n_mfcc=40,
                                     n_fft=self.window_size,
                                     hop_length=int(self.window_size/2))

        mean_mfcc = np.mean(mfccs, axis=1)
        std_mfcc = np.std(mfccs, axis=1)
        mean_diff = np.mean(np.diff(mfccs, axis=1), axis=1)

        mfcc_features = np.concatenate([mean_mfcc, std_mfcc, mean_diff])

        return mfcc_features


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