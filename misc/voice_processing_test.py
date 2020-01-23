"""
This is a very first attempt in creating an audio processing framework for dealing with voices
for the h-game project

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

# Audio processing modules
import soundfile as sf
import python_speech_features as psf

input_path = r'C:\game_data\game_0704\audio\voice'
output_path = r'C:\game_data\game_0704\audio'

files_df = pd.DataFrame(columns=['filename', 'duration'])

t0 = time.time()
for (dirpath, dirnames, filenames) in os.walk(input_path):

    # Add all the filenames at once
    files_df['filename'] = filenames
    for i, row in files_df.iterrows():
        data, fs = sf.read(os.path.join(input_path, row['filename']))
        row['duration'] = len(data)/fs
        #print(f'[{i+1}/{len(filenames)}] {file} : {length:.2f}')
t1 = time.time()
print(f'Elapsed time = {t1-t0:.2f} seconds')

output_fpath = os.path.join(output_path, 'voices_dataframe.csv')
files_df.to_csv(output_fpath, header=True)

#  ============= PREMATURE RETURN =========

"""
mcff = psf.mfcc(signal=data, samplerate=fs)

plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(mcff[:, 0:3])
plt.show()
g = 0

"""