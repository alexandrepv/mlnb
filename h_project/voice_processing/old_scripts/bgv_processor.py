import os
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Signal processing modules
from scipy.signal import hilbert, chirp
from scipy import signal
import soundfile as sf
import python_speech_features as psf

def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = np.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1],
                 inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = np.ones(windowLen, 'd')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[windowLen:-windowLen+1]

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


def split_audio_into_intervals(raw_signal, fs, spectogram_thres=0.0012):

    """

    :param raw_signal:
    :param fs:
    :param threshold:
    :return: list of dictionaries
    """

    # Define constants
    windows_size = 512
    spectogram_scale = 750      # TODO: Make these two numbers one!

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

    intervals_fixed = []
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            if intervals[i - 1][1] - intervals[i][0] == 1:
                intervals_fixed.append([intervals[i - 1][0], intervals[i - 1][1]])
            else:
                intervals_fixed.append(intervals[i])

    offset_frames = windows_size / 2
    intervals_fixed_np = np.array(intervals)
    intervals_frames = np.zeros_like(intervals_fixed_np)
    intervals_frames[:, 0] = frames[intervals_fixed_np[:, 0]] - offset_frames
    intervals_frames[:, 1] = frames[intervals_fixed_np[:, 1]] - offset_frames

    intervals_visual = np.zeros((num_samples,))
    for inter in intervals_frames:
        intervals_visual[inter[0]:inter[1]] = 1

    plt.figure()
    plt.plot(raw_signal)
    plt.plot(spectogram_x_axis, spectogram_amplitudes)
    plt.plot(spectogram_x_axis, spectogram_valid_windows * 0.075)
    plt.plot(intervals_visual * 0.05)
    plt.legend(['Raw Signal',
                'Spectrogram',
                'Spectogram Valid Interval',
                'Actual Intervals'])
    plt.show()

def process(input_fpath):

    raw_data, fs = sf.read(input_fpath)
    num_samples = len(raw_data)
    duration_sec = num_samples/fs

    split_audio_into_intervals(raw_data, fs)


    f, t, Zxx = signal.stft(raw_data, fs, nperseg=512)

    freqs, times, Sx = signal.spectrogram(raw_data, fs=fs, window='hanning',
                                          nperseg=512, noverlap=256,
                                          detrend=False, scaling='spectrum')

    plt.figure()
    #plt.imshow(Sx, cmap='nipy_spectral')
    plt.plot(np.sum(Sx, axis=0))
    plt.show()

    stft_abs = np.absolute(Zxx)
    stft_abs_1d = np.reshape(stft_abs, stft_abs.shape[0]*stft_abs.shape[1])
    stft_abs_1d = stft_abs_1d[stft_abs_1d >= 0]
    stft_abs_sum = np.sum(stft_abs, axis=0)

    mcff = psf.mfcc(signal=raw_data, samplerate=fs)

    analytic_signal = hilbert(raw_data)
    amplitude_envelope = np.abs(analytic_signal)

    smooth_envelope = smoothMovingAvg(amplitude_envelope)

    time_raw = np.arange(num_samples)/fs
    time_fft = np.linspace(0, duration_sec, Zxx.shape[1])

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time_raw, raw_data)
    plt.xlim(0, duration_sec)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f, stft_abs, vmin=0, cmap='nipy_spectral')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(2, 1, 1)
    plt.plot(time_fft, stft_abs_sum)
    plt.xlim(0, duration_sec)
    plt.subplot(2, 1, 1)
    plt.plot(time_raw, smooth_envelope)
    plt.xlim(0, duration_sec)
    plt.show()

    intervals  = split_audio_into_intervals()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_raw, raw_data)
    plt.xlim(0, duration_sec)
    plt.subplot(3, 1, 2)
    plt.plot(time_raw, amplitude_envelope)
    plt.plot(time_raw, smooth_envelope)
    plt.xlim(0, duration_sec)
    plt.subplot(3, 1, 3)
    plt.imshow(np.transpose(mcff))
    plt.title(f'Raw signal, fs = {fs} Hz')
    plt.xlabel('Time (seconds)')
    plt.show()

