import argparse
import numpy as np
import os
import time

import matplotlib
matplotlib.use('Agg')

import dnae_scripts.isfet_classifier_v3 as isfet_classifier_v3
import dnae_scripts.default as default

# This is to prevent stupid JSON library from complaining about bloody integers
# from: https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
def convert(number):
    if isinstance(number, np.int64): return int(number)
    raise TypeError

if __name__ == '__main__':

    # =========================================================================
    #                           Input Arguments
    # =========================================================================

    parser = argparse.ArgumentParser()

    parser.add_argument('training_h5_data_folder',
                        type=str,
                        help='Folder where all the labelled signals are located')
    parser.add_argument('output_folder',
                        type=str,
                        default='',
                        help='Output directory where you want to save the model recipe and model file. If left BLANK,'
                             'it will be the same as the dataset')
    parser.add_argument('--layers',
                        default=default.NN_LAYERS,
                        type=int,
                        nargs='+',
                        help='Maximum number of decimal places on the log separated list of learning rates')
    parser.add_argument('--optimizer',
                        type=str,
                        default=default.NN_OPTIMIZER,
                        help='Optimizer to be used to test the architecture')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=default.NN_LEARNING_RATE,
                        help='Optimizer to be used to test the architecture')
    parser.add_argument('--epochs',
                        type=int,
                        default=default.NN_EPOCHS,
                        help='Output directory where you want to save the model recipe and model file. If left BLANK,'
                             'it will be the same as the dataset')

    args = parser.parse_args()

    if len(args.output_folder) != 0:
        if not os.path.exists(args.output_folder):
            raise Exception(f"[ERROR] Output directory '{args.output_folder}' doesn't exist")
    else:
        args.output_folder = args.training_h5_data_folder

    # Create new output folder using the timestamp
    args.output_folder = os.path.join(args.output_folder, f'training_session_{time.strftime("%Y_%m_%d_%Hh%Mm%Ss")}')
    os.makedirs(args.output_folder, exist_ok=True)

    # Train network
    icv3 = isfet_classifier_v3.IsfetClassifierV3()
    icv3.train_model_from_h5_folder(h5_input_folder=args.training_h5_data_folder,
                                    output_folder=args.output_folder,
                                    layers=args.layers,
                                    optimizer=args.optimizer,
                                    lr=args.learning_rate,
                                    epochs=args.epochs)