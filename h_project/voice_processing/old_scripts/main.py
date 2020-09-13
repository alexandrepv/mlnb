import argparse
import h_project.voice_processing.old_scripts.game_database as game_database

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.interpolate import splev, splrep
    import numpy as np

    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    spl = splrep(x, y)
    x2 = np.linspace(0, 10, 200)
    y2 = splev(x2, spl)
    plt.plot(x, y, 'o', x2, y2)
    plt.show()

    # Step 0) Parse arguments
    parser = argparse.ArgumentParser(description='Process all voice files inside a folder')
    parser.add_argument('voice_folder',
                        type=str,
                        help='Folder containing all the files you want to process')
    parser.add_argument('xml_filepath',
                        type=str,
                        help='XML file containing the description of the game')
    parser.add_argument('output_filepath',
                        type=str,
                        help='Output JSON file')

    args = parser.parse_args()

    xml_fpath = r"I:\Hentai_Game_Project_Resources\game_resource_archive\game_0708\game.xml"
    audacity_fpath = r'D:\Dropbox\Projects\Hentai Game Project\Development\voice_processing\manually_labelled_files\game_0708\audacity_labels.json'
    db_filepath = r'D:\Dropbox\Projects\Hentai Game Project\Development\voice_processing\manually_labelled_files\game_0708\game_db.json'

    db = game_database.GameDB(xml_fpath)
    #db.load_audacity_json(audacity_fpath)
    #db.save(db_filepath)
    db.load(db_filepath)

    features = db.extract_audio_features()
