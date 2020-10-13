import argparse

import h_project.voice_processing.data_exploration as data_exploration
import h_project.voice_processing.audio_segmentation.label_loader as label_loader
from h_project.voice_processing.audio_segmentation.audio_segmenter import AudioSegmenter
from h_project.voice_processing.character_voice_organizer import CharacterVoiceOrganizer

import librosa

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process all voice files inside a folder')

    parser.add_argument('audio_folder',
                        type=str,
                        help='Folder where the audio files are')
    parser.add_argument('labels_folder',
                        type=str,
                        help='Folder where the Audacity label files are')
    parser.add_argument('--predict_audio_folder',
                        type=str,
                        help='Folder for testing the prediction on new audio files')

    args = parser.parse_args()

    audio_seg = AudioSegmenter()

    voice_organizer = CharacterVoiceOrganizer()
    #voice_organizer.calculate_folder_means_and_std(root_folder=args.audio_folder, audio_segmenter=audio_seg)

    voice_organizer.test_different_thresholds(root_folder=args.audio_folder,
                                              audio_segmenter=audio_seg,
                                              fs=44100)

    #labels = label_loader.load_audacity_labels(r'D:\data\audio_bgv_labels\game_0708\audacity_amu_bgv_0013.txt')

    #clustering = data_exploration.AudioFileClustering()
    #clustering.cluster_avg_per_window([r'D:\data\audio_bgv\game_0044',
    #                                   r'D:\data\__game_0708',
    #                                   r'D:\data\audio_bgv\game_0049',
    #                                   r'D:\data\audio_bgv\game_0105',
    #                                   r'D:\data\audio_bgv\game_0867',
    #                                   r'D:\data\audio_bgv\game_1026'])

    #audacity_mgr = audacity_label_manager.SmartAudioSegmenter()
    #utils.find_all_xml_files_and_save_as_csv(root_folder='I:\Hentai_Game_Project_Resources\game_resource_archive',
    #                                         output_csv_fpath=r'D:\h_project\data\game_db.csv')

    """
    fpath = r'D:\h_project\data\audio\game_0708\audio\voice_bgv\amu_bgv_0014.ogg'
    audacity_mgr.cluster_audio_features(audio_fpath=fpath,
                                        num_clusters=8,
                                        num_quitest_cluster=1,
                                        num_mfcc_features=13)

    audacity_mgr.load_training_data(audio_folder=args.audio_folder,
                                    labels_folder=args.labels_folder,
                                    max_files=-1,)
    audacity_mgr.train_model_voice_and_silence(n_estimators=128)
    audacity_mgr.predict_model_void_and_silence(args.predict_audio_folder)
    """
    """for audio in audacity_mgr.audio_list:
        mfccs = librosa.feature.mfcc(audio['signal'], audio['fs'], n_mfcc=13)

        # Figure out which window each label sits

        plt.subplot(2, 1, 1)
        plt.title(audio['filename'])
        plt.imshow(np.clip(mfccs, 0, 1000), aspect='auto')
        plt.subplot(2, 1, 2)
        plt.plot(audio['signal'])
        for index, label in audio['labels'].iterrows():
            plt.axvline(x=label['start_frame']+1, linewidth=1.5, color='g')
            plt.axvline(x=label['stop_frame']-1, linewidth=1.5, color='r')
        plt.xlim(0, len(audio['signal']))
        plt.tight_layout()
        plt.show()"""

    # Load audio signal
    #S = librosa.feature.melspectrogram(y=raw_data, sr=fs, n_mels=13)
    #mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))


    #librosa.display.specshow(mfccs, x_axis='time')


    """audio_seg = audio_segmenter.AudioSegmenter(fs=fs)

    #audio_seg.process_with_librosa(raw_signal=raw_data_2)

    audio_seg.process_with_avg(raw_signal=raw_data,
                               threshold_bottom=0.015,
                               threshold_top=0.01)

    threshold = 1.0  #0.000011"""
    #audio_seg.process_with_spectogram(raw_data_2, threshold=threshold)






