import pandas as pd


def add_numbers_to_audacity_label_files(filepath):

    audacity_labels_df = pd.read_csv(filepath,
                                     names=['time_start', 'time_stop', 'label'],
                                     delimiter='\t')

    for idx, row in audacity_labels_df.iterrows():

        if audacity_labels_df.loc[idx, 'label'][0] != '[':
            audacity_labels_df.loc[idx, 'label'] = f'[{idx}]' + audacity_labels_df.loc[idx, 'label']

    audacity_labels_df.to_csv(filepath,
                              header=False,
                              index=False,
                              sep='\t',
                              float_format='%.6f')


filepath = r'G:\Dropbox\Projects\Hentai Game Project\Development\voice_processing\manually_labelled_files\game_0708\test\game_0708_ran_bgv_0016.txt'


add_numbers_to_audacity_label_files(filepath)