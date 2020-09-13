import argparse
import h_project.voice_processing.old_scripts.bgv_processor as bgv_processor

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process all voice files inside a folder')
    parser.add_argument('voice_fpath',
                        type=str,
                        help='Audio voice file that you want to process')

    args = parser.parse_args()

    bgv_processor.process(args.voice_fpath)