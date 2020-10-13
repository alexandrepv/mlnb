import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AudioSegmenter:

    def process_auto(self, signal_raw):

        pass

    def process(self, signal_raw, window_size=256, silence_threshold=0.00075, debug=False):

        """
        This segments the audio and returns invervals in frames.
        For now, only non-overlapting windows

        :param signal:
        :param parameters:
        :return:
        """

        # =============================================================
        #            Calculate Mean and Std for each window
        # =============================================================

        signal_abs = np.abs(signal_raw)
        num_frames = signal_raw.size
        num_steps = num_frames // window_size + 1
        if num_frames % window_size != 0:
            num_steps += 1
        windows_indices = np.arange(0, num_steps) * window_size
        windows_indices[-1] = num_frames

        windows_means = np.ndarray((windows_indices.size-1, ), dtype=np.float32)
        for j in range(windows_indices.size - 1):
            a = windows_indices[j]
            b = windows_indices[j + 1]
            windows_means[j] = np.mean(signal_abs[a:b])
        positive_windows = windows_means > silence_threshold

        # =============================================================
        #    Adjust valid windows and determine intervals in frames
        # =============================================================

        valid_intervals_windows = self.get_valid_intervals(positive_windows)
        valid_intervals_frames = np.clip(np.array(valid_intervals_windows) * window_size, 0, num_frames)

        valid_signal = np.zeros((num_frames,), dtype=np.bool)
        for i in range(valid_intervals_frames.shape[0]):
            valid_signal[valid_intervals_frames[i, 0]:valid_intervals_frames[i, 1]] = True

        if debug:
            plt.plot(signal_raw)
            plt.plot(valid_signal)
            plt.step(windows_indices[:-1], windows_means, '.-', where='post')
            plt.step(windows_indices[:-1], positive_windows*0.5, '.-', where='post')
            plt.hlines(y=silence_threshold, xmin=0, xmax=num_frames, colors='r')
            plt.show()

        return valid_intervals_frames

    def get_valid_intervals(self, input_mask, min_size=3):

        """
        Finds intervals in the inputmask of continuous True

        :param input_mask: boolean mask of valid intervals
        :param min_size: minimum size of groups of True
        :return:
        """

        num_windows = len(input_mask)
        new_interval = True
        current_interval_start = 0
        intervals = []

        # Convert all continuous True windows into an interval
        for i in range(num_windows):

            # Start a new interval
            if new_interval and input_mask[i]:
                new_interval = False
                current_interval_start = i

            # Append to current interval
            if not new_interval and not input_mask[i]:
                new_interval = True
                intervals.append([current_interval_start, i])

        # Only return intervals greater or equal to the valid minimum size
        valid_intervals = [itv for itv in intervals if itv[1]-itv[0] >= min_size]

        return valid_intervals

    def get_mean_std_from_intervals(self, signal_raw, frame_intervals):

        """
        returns the mean and std from absolute signal from each of the intervals

        :param signal_raw:
        :param intervals: numpy ndarray (M_intervals, 2) -> [start_frame, stop_frame[
        :return:
        """
        num_intervals = frame_intervals.shape[0]

        intervals_mean = np.ndarray((num_intervals,), dtype=np.float32)
        intervals_std = np.ndarray((num_intervals,), dtype=np.float32)

        for i in range(num_intervals):
            a = frame_intervals[i, 0]
            b = frame_intervals[i, 1]
            intervals_mean[i] = np.mean(signal_raw[a:b])
            intervals_std[i] = np.std(signal_raw[a:b])

        return intervals_mean, intervals_std
