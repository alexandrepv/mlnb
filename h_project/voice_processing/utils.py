import glob
import os
import re
import shutil
import xmltodict
import numpy as np
import pandas as pd

def get_all_subfolders_in_folder(root_folder):
    all_folders_in_root = [os.path.join(root_folder, o) for o in os.listdir(root_folder) if
                           os.path.isdir(os.path.join(root_folder, o))]

    return all_folders_in_root

def copy_files_to_new_game_folders(list_of_game_dicts, destination_root_directory):

    """
    This will also convert subfolders into
    :param list_of_game_dicts:
    :return:
    """

    print('[ Copying Files ]')
    for i, game_dict in enumerate(list_of_game_dicts):

        print(f' {i+1}/{len(list_of_game_dicts)} ')
        game_label = list(game_dict.keys())[0]

        game_folder = os.path.join(destination_root_directory, game_label)
        os.makedirs(game_folder, exist_ok=True)

        for filepath in game_dict[game_label]:

            matches = re.search(r'(.*)audio\\voice_bgv\\(.*)', filepath)
            _, filename = os.path.split(filepath)
            new_filename = matches[2].replace('\\', '_')

            new_filepath = os.path.join(game_folder, new_filename)
            shutil.copyfile(filepath, new_filepath)

def find_all_bgv_files(root_folder):

    all_folders_in_root = [os.path.join(root_folder, o) for o in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, o))]
    all_games_labels = []
    games_with_voice_bgv_folders = []
    valid_audio_extensions = ['.ogg', '.wav', '.mp3', '.ogv', '.ovk']

    # Select only valid game folder label
    for game_folder in all_folders_in_root:
        _, last_bit = os.path.split(game_folder)
        if 'game_' in last_bit:
            all_games_labels.append(last_bit)

    for i, game_label in enumerate(all_games_labels):
        #print(f' {i+1}/{len(all_games_labels)} {game_label}', end='')

        game_folder = os.path.join(root_folder, game_label)
        game_bgv_folder = os.path.join(game_folder, 'audio', 'voice_bgv')
        if os.path.exists(game_bgv_folder):

            all_bgv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(game_bgv_folder)
                                      for f in filenames if os.path.splitext(f)[1].lower() in valid_audio_extensions]

            #for root, subdirs, files in os.walk(game_bgv_folder):
            #    if len(subdirs) != 0:
            #        print(' - Subdirectory detected', end='')

            new_game_dict = {game_label:all_bgv_files}
            games_with_voice_bgv_folders.append(new_game_dict)
        #print('')

    return games_with_voice_bgv_folders

def find_all_xml_files_and_save_as_csv(root_folder, output_csv_fpath):


    all_folders_in_root = [os.path.join(root_folder, o) for o in os.listdir(root_folder) if
                           os.path.isdir(os.path.join(root_folder, o))]
    all_games_labels = []

    # Select only valid game folder label
    for game_folder in all_folders_in_root:
        _, last_bit = os.path.split(game_folder)
        if 'game_' in last_bit:
            all_games_labels.append(last_bit)

    columns = ['label',
               'release_date',
               'developer',
               'title_english',
               'title_romanji',
               'title_japanese',
               'notes']

    games_df = pd.DataFrame(columns=columns, dtype=str)
    games_df['label'] = all_games_labels

    print('[ Analysing Game folders]')
    for i, game_label in enumerate(all_games_labels):


        game_xml_fpath = os.path.join(root_folder, game_label, 'game.xml')

        with open(game_xml_fpath, 'r', encoding='cp932', errors='ignore') as file:
            game_dict = xmltodict.parse(file.read(), process_namespaces=True)

            print(f'{i + 1}/{len(all_games_labels)} {game_label}')

            # TITLES
            if type(game_dict['game_resource']['title']) is list:
                titles = game_dict['game_resource']['title']
            else:
                titles = [game_dict['game_resource']['title']]
            for title in titles:
                language = title['@language']
                if '#text' in title.keys():
                    text = title['#text']
                else:
                    text = ''
                if language == 'english':
                    title_english = text
                if language == 'romanji':
                    title_romanji = text
                if language == 'japanese':
                    title_japanese = text

            # RELEASE DATE
            year = game_dict['game_resource']['release_date']['@year']
            month = game_dict['game_resource']['release_date']['@month']
            day = game_dict['game_resource']['release_date']['@day']
            release_date = f'{year}-{month}-{day}'

            # DEVELOPER
            if 'developer' in game_dict['game_resource']:
                developer = game_dict['game_resource']['developer']
            else:
                developer = ''

            # NOTES
            if 'notes' in game_dict['game_resource']:
                notes = game_dict['game_resource']['notes']
            else:
                notes = ''

            if notes is None:
                notes = ''

            fixed_notes = remove_invalid_lines_from_notes(notes)

            # Update dataframe
            games_df.at[i, 'release_date'] = release_date
            games_df.at[i, 'developer'] = developer
            games_df.at[i, 'title_english'] = title_english
            games_df.at[i, 'title_romanji'] = title_romanji
            games_df.at[i, 'title_japanese'] = title_japanese
            games_df.at[i, 'notes'] = fixed_notes

    games_df.to_csv(output_csv_fpath, sep=',', index=False, index_label=False)

def remove_invalid_lines_from_notes(input_notes):

    # removing unwanted lines from notes
    lines = input_notes.split('\n')
    invalid_lines_mask = np.zeros((len(lines),), dtype=np.bool)
    empty_lines_mask = np.zeros((len(lines),), dtype=np.bool)
    for i, line in enumerate(lines):
        line = line.replace('\t','')
        if line.lower().startswith('voiced by'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('voicedby'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('main character'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('maincharacter'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('side character'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('sidecharacter'):
            invalid_lines_mask[i] = True
        if line.lower().startswith('protagonist'):
            invalid_lines_mask[i] = True
        if line == '':
            empty_lines_mask[i] = True
    valid_lines_indices = np.where(np.logical_not(invalid_lines_mask))[0]
    lines_np = np.array(lines)
    notes = str(' '.join(lines_np[valid_lines_indices]))

    return notes