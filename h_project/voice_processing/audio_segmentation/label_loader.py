import pandas as pd
import numpy as np

import h_project.voice_processing.default as default


def load_audacity_frame_interval_labels(labels_fpath, fs=44100):

    """
    Load labels from audacitty txt files. The files are int he format

    [start_time  stop_time label]
    0.2344      0.2517    inhale
    0.3141      0.3856    exhale
    ...

    This function returns a pandas dataframe in the format [label start_frame stop_frame] by convertint the time
    stamp into a frame


    :param labels_fpath_list:
    :param fs:
    :return:
    """

    audacity_labels_df = pd.read_csv(labels_fpath, delimiter='\t', names=['start_time', 'stop_time', 'label'])

    frame_interval_labels_df = pd.DataFrame(columns=[default.LABEL_DATAFRAME_KEY_LABEL,
                                      default.LABEL_DATAFRAME_KEY_START_FRAME,
                                      default.LABEL_DATAFRAME_KEY_STOP_FRAME])
    frame_interval_labels_df[default.LABEL_DATAFRAME_KEY_LABEL] = audacity_labels_df['label']
    frame_interval_labels_df[default.LABEL_DATAFRAME_KEY_START_FRAME] = np.round(audacity_labels_df['start_time'].values * fs).astype(np.int)
    frame_interval_labels_df[default.LABEL_DATAFRAME_KEY_STOP_FRAME] = np.round(audacity_labels_df['stop_time'].values * fs).astype(np.int)

    return frame_interval_labels_df