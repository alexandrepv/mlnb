import os

import json
import time
import numpy as np
import pandas as pd
import glob
import warnings
import h5py

import dnae_scripts.default as default

# Tensorflow Modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(f'GPU is available: {tf.test.is_gpu_available()}')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class IsfetClassifierV3:

    def __init__(self):

        # Input Data Variables
        self.input_dimensions = int(default.SIGNAL_EXTR_NUM_SAMPLES_PER_FLOW)

        # Preprocessing Variables
        self.ref_percentile = float(default.SIGNAL_EXTR_GLOBAL_REF_PERCENTILE)
        self.norm_min_value = float(default.SIGNAL_EXTR_NORM_MIN_VALUE)
        self.norm_max_value = float(default.SIGNAL_EXTR_NORM_MAX_VALUE)

        # Neural Network Variables
        self.model = None
        self.layers = None
        self.input_dimensions = None
        self.learning_rate = None
        self.optimizer = None

    def load_model_recipe(self, json_fpath):

        """
        Loads the JSON recipe that instructs how to preprocess the data and where to load the Tensoflow model from
        :param json_fpath:
        :return:
        """

        with open(json_fpath) as file:
            recipe_dict = json.load(file)

            self.layers = recipe_dict[default.NN_RECIPE_KEY_LAYERS]
            self.optimizer = recipe_dict[default.NN_RECIPE_KEY_OPTIMIZER]
            self.learning_rate = recipe_dict[default.NN_RECIPE_KEY_LEARNING_RATE]
            self.norm_min_value = recipe_dict[default.NN_RECIPE_KEY_NORM_MIN_VALUE]
            self.norm_max_value = recipe_dict[default.NN_RECIPE_KEY_NORM_MAX_VALUE]
            self.input_dimensions = recipe_dict[default.NN_RECIPE_KEY_INPUT_DATA_DIMENSIONS]
            self.ref_percentile = recipe_dict[default.NN_RECIPE_KEY_REF_PERCENTILE]

            directory, _ = os.path.split(json_fpath)
            fpath = os.path.join(directory, recipe_dict[default.NN_RECIPE_KEY_MODEL_H5_FILENAME])
            self.model = tf.keras.models.load_model(fpath)

    def train_model_from_h5_folder(self,
                                   h5_input_folder,
                                   output_folder,
                                   layers=default.NN_LAYERS,
                                   lr=default.NN_LEARNING_RATE,
                                   epochs=default.NN_EPOCHS,
                                   batch_size=10000,
                                   optimizer=default.NN_OPTIMIZER,
                                   norm_min_value=default.SIGNAL_EXTR_NORM_MIN_VALUE,
                                   norm_max_value=default.SIGNAL_EXTR_NORM_MAX_VALUE,
                                   validation_split=default.NN_VALIDATION_SPLIT,
                                   early_stop_patience=default.NN_EARLY_STOPPING_PATIENCE,
                                   verbose=True):

        signals, labels, ref_percentile = self.load_signals_and_labels_from_all_h5_files(h5_folder_path=h5_input_folder,
                                                                                         verbose=verbose)

        self.layers = layers
        self.learning_rate = lr
        self.optimizer = optimizer
        self.input_dimensions = signals.shape[1]
        self.ref_percentile = ref_percentile
        self.norm_min_value = norm_min_value
        self.norm_max_value = norm_max_value

        self.normalize_signal(signals,min_value=self.norm_min_value, max_value=self.norm_max_value)

        mid_layers = self.get_mid_layers(layers)

        # Create optimizer
        if optimizer == 'adam':
            optimizer_obj = Adam(lr=lr)
        elif optimizer == 'sgd':
            optimizer_obj = SGD(lr=lr)
        else:
            raise Exception(f"[ERROR] Optimizer '{optimizer}' not available")

        tf.keras.backend.clear_session()

        # Create new neural network model
        model = Sequential()
        model.add(Dense(self.layers[0],
                             input_shape=(self.input_dimensions,),
                             activation=default.NN_MID_LAYERS_ACTIVATION_FUNCTION))
        for i in range(1, len(mid_layers)):
            model.add(Dense(mid_layers[i],
                                 input_shape=(self.input_dimensions,),
                                 activation=default.NN_MID_LAYERS_ACTIVATION_FUNCTION))
        model.add(Dense(1, activation=default.NN_FINAL_LAYER_ACTIVATION_FUNCTION))
        model.compile(optimizer=optimizer_obj,
                           loss=default.NN_LOSS_FUNCTION,
                           metrics=default.NN_METRICS)

        # Training
        if verbose:
            print('\n[ Training Model ]', flush=True)
            print(f' > Layers : {self.layers}')
            print(f' > Optimizer : {self.optimizer}')
            print(f' > Learning Rate : {self.learning_rate}')
        training_result = model.fit(signals,
                                    labels,
                                    epochs=epochs,
                                    validation_split=validation_split,
                                    batch_size=batch_size,
                                    verbose=True)
        if verbose:
            print(' > Done')

        # Save model and results
        json_fpath = os.path.join(output_folder, 'model_recipe.json')
        self.save_model_recipe(json_fpath, model=model)
        training_history_df = self.save_training_results(output_folder=output_folder,
                                                         keras_results=training_result)

        del(model)
        del(signals)
        del(labels)

        return training_history_df

    def save_training_results(self, output_folder, keras_results):

        # Save training history
        fpath = os.path.join(output_folder, 'training_results.csv')
        training_history_df = pd.DataFrame(keras_results.history)
        training_history_df.index += 1  # Let's counting from one just this once :)
        training_history_df.to_csv(fpath, sep=',', index_label='epoch', index=True)

        # Save 'accuracy' plot
        accuracy_keys = np.sort([key for key in keras_results.history.keys() if 'accuracy' in key])
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plt.plot(training_history_df.index, (training_history_df[accuracy_keys]*100))
        plt.title('Training Accuracy', fontdict={'size': default.PLOT_TITLE_SIZE})
        plt.xlabel('Epochs', fontsize=default.PLOT_AXES_LABEL_SIZE)
        plt.ylabel('Accuracy (%)', fontsize=default.PLOT_AXES_LABEL_SIZE)
        plt.xlim(1, training_history_df.index.size)
        plt.legend(accuracy_keys)
        plt.grid(True)
        fpath = os.path.join(output_folder, 'training_accuracy.png')
        plt.savefig(fpath, dpi=160)
        plt.tight_layout()
        plt.close(fig)

        # Save 'loss' plot
        loss_keys = np.sort([key for key in keras_results.history.keys() if 'loss' in key])
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plt.plot(training_history_df.index, training_history_df[loss_keys])
        plt.title('Training Loss', fontdict={'size': default.PLOT_TITLE_SIZE})
        plt.xlabel('Epochs', fontsize=default.PLOT_AXES_LABEL_SIZE)
        plt.ylabel('Loss', fontsize=default.PLOT_AXES_LABEL_SIZE)
        plt.xlim(1, training_history_df.index.size)
        plt.legend(loss_keys)
        plt.grid(True)
        fpath = os.path.join(output_folder, 'training_loss.png')
        plt.savefig(fpath, dpi=160)
        plt.tight_layout()
        plt.close(fig)

        return training_history_df

    def load_signals_and_labels_from_all_h5_files(self, h5_folder_path, shuffle_data=True, verbose=True):

        """
        Loads the labelled signals from all the .h5 files. Be careful as this will load ALL the data into memory,
        so beware of how much data you have in that folder.
        TODO: Expand this solution to do pre-fetching data for training

        :param folder_path:
        :return:
        """

        # Load all labelled signals from folder
        fpath_target_str = os.path.join(h5_folder_path, '*.h5')
        h5_signal_files = glob.glob(fpath_target_str)

        if len(h5_signal_files) == 0:
            raise Exception(f"[ERROR] There are no '.h5' files in '{h5_folder_path}'")

        # Step 1) Allocate the exact amount of memory we'll need to load all the training data
        num_isfets = 0
        input_dimensions_list = []
        ref_percentile_list = []
        for h5_fpath in h5_signal_files:
            h5_file = h5py.File(h5_fpath, 'r')
            ref_percentile_list.append(h5_file.attrs[default.H5_SIGNALS_KEY_REF_PERCENTILE])
            for group_key in list(h5_file.keys()):
                num_isfets += h5_file[group_key][default.H5_SIGNALS_GROUP_KEY_SIGNALS].shape[0]
                input_dimensions_list.append(h5_file[group_key][default.H5_SIGNALS_GROUP_KEY_SIGNALS].shape[1])
            h5_file.close()

        # Step 2) Make sure all data has the same input dimesnions and ref percentile
        unique_input_dimensions = np.unique(input_dimensions_list)
        if unique_input_dimensions.size > 1:
            raise Exception('[ERROR] Not all training data files contain flows with the same number of dimensions.')
        input_data_dimensions = int(unique_input_dimensions[0])
        unique_ref_percentiles = np.unique(ref_percentile_list)
        if unique_ref_percentiles.size > 1:
            raise Exception('[ERROR] Not all training data files contain flows with the same number reference percentiles.')
        ref_percentile = int(unique_ref_percentiles[0])

        # Step 3) Load all the data
        signals = np.ndarray((num_isfets, input_data_dimensions), dtype=np.float32)
        labels = np.ndarray((num_isfets,), dtype=np.int32)
        a = 0
        if verbose:
            print('\n[ Loading Training Data ]', flush=True)
        for i, h5_fpath in enumerate(h5_signal_files):
            _, filename = os.path.split(h5_fpath)
            if verbose:
                print(f' > {i + 1}/{len(h5_signal_files)} {filename}', end='', flush=True)
            h5_file = h5py.File(h5_fpath, 'r')
            for group_key in list(h5_file.keys()):
                b = a + h5_file[group_key][default.H5_SIGNALS_GROUP_KEY_SIGNALS].shape[0]
                signals[a:b, :] = h5_file[group_key][default.H5_SIGNALS_GROUP_KEY_SIGNALS]
                labels[a:b] = h5_file[group_key][default.H5_SIGNALS_GROUP_KEY_LABELS]
                a = np.copy(b)
            h5_file.close()
            if verbose:
                print(f' - OK', flush=True)

        # Step 4) Shuffle data in-place (if selected)
        if shuffle_data:
            if verbose:
                print('\n[ Shuffling Training Data ]', flush=True)
            rstate = np.random.RandomState(default.SHUFFLING_DATA_RANDOM_STATE)
            rstate.shuffle(signals)
            rstate = np.random.RandomState(default.SHUFFLING_DATA_RANDOM_STATE)
            rstate.shuffle(labels)
            if verbose:
                print(' > Done')

        return signals, labels, ref_percentile

    def save_model_recipe(self, json_fpath, model):

        directory, recipe_filename = os.path.split(json_fpath)
        recipe_label, _ = os.path.splitext(json_fpath)

        model_h5_filename = 'model.h5'

        # Create names for both recipe and Tensorflow model files
        recipe_fpath = os.path.join(directory, f'{recipe_label}.json')
        model_h5_fpath = os.path.join(directory, model_h5_filename)

        # Create recipe
        model_recipe_dict = {default.NN_RECIPE_KEY_MODEL_H5_FILENAME: model_h5_filename,
                             default.NN_RECIPE_KEY_LAYERS: self.layers,
                             default.NN_RECIPE_KEY_OPTIMIZER: self.optimizer,
                             default.NN_RECIPE_KEY_LEARNING_RATE: self.learning_rate,
                             default.NN_RECIPE_KEY_NORM_MIN_VALUE: self.norm_min_value,
                             default.NN_RECIPE_KEY_NORM_MAX_VALUE: self.norm_max_value,
                             default.NN_RECIPE_KEY_INPUT_DATA_DIMENSIONS: self.input_dimensions,
                             default.NN_RECIPE_KEY_REF_PERCENTILE: self.ref_percentile}

        # Save recipe
        with open(recipe_fpath, 'w') as file:
            json.dump(model_recipe_dict, file, indent=2)

        # Save recipe - including tensorlfow model
        model.save(model_h5_fpath)

    def check_input_arguments(self, args):

        # Check input arguments
        args.dnb5_fpath = utils_string.convert_to_linux_azure_path(args.dnb5_fpath)
        args.model_recipe_json_fpath = utils_string.convert_to_linux_azure_path(args.model_recipe_json_fpath)

        if len(args.runid) == 0:
            args.runid = utils_string.get_runid_from_string(args.dnb5_fpath)

        if not os.path.isfile(args.dnb5_fpath):
            raise Exception(f"[ERROR] Cannot open '{args.dnb5_fpath}'")
        if not os.path.isfile(args.model_recipe_json_fpath):
            raise Exception(f"[ERROR] Cannot open '{args.model_recipe_json_fpath}'")

        if len(args.spotmap_xlsx_fpath) != 0:
            args.spotmap_xlsx_fpath = utils_string.convert_to_linux_azure_path(args.spotmap_xlsx_fpath)
            if not os.path.exists(args.spotmap_xlsx_fpath):
                raise Exception(f"[ERROR] Cannot open '{args.spotmap_xlsx_fpath}' doesn't exist.")

        if len(args.output_directory) != 0:
            if not os.path.exists(args.output_directory):
                raise Exception(f"[ERROR] Output directory '{args.output_directory}' doesn't exist.")
        else:
            args.output_directory, _ = os.path.split(args.dnb5_fpath)
        args.output_directory = os.path.join(args.output_directory, f'{args.runid}_Isfet_Classifier_v3')
        os.makedirs(args.output_directory, exist_ok=True)

    def normalize_signal(self,signals,
                         min_value=default.SIGNAL_EXTR_NORM_MIN_VALUE,
                         max_value=default.SIGNAL_EXTR_NORM_MAX_VALUE):

        """
        Normalises all individual signals from "signals" (row per row) IN PLACE!S

        :param signal: numpy ndarray (n_isfets, m_dimensions)
        :param min_value: float
        :param max_value: float
        """

        np.clip(signals, min_value, max_value, out=signals)
        signals -= min_value
        signals /= (max_value - min_value)

    def generate_parameters_string_from_args(self, args):

        """
        This is just a simple way to unify report presentation across multiple projects
        """

        # Create a complete report of all the relevant parameters
        parameters_str = f'==========[ Isfet Classifier v3 ]==========\n'
        parameters_str += '\n[ Input Parameters ]\n'
        for arg in vars(args):
            parameters_str += f' > {arg} : {getattr(args, arg)}\n'

        with open(args.model_recipe_json_fpath) as file:

            model_recipe_dict = json.load(file)
            parameters_str += '\n[ Model Parameters ]\n'
            for key in model_recipe_dict.keys():
                parameters_str += f' > {key} : {model_recipe_dict[key]}\n'

        return parameters_str

    def get_mid_layers(self, layers: list):

        """

        :param layers: List of intergers representing the number of neuros per layer.
        :return: Validaded list of intergers. If the last number is 1, it is removed
        """

        if layers is None:
            raise Exception('[ERROR] Layers cannot be None')

        if len(layers) == 0:
            raise Exception('[ERROR] Layers be empty')

        # Remove any layers with 1 neuron
        layers_np = np.array(layers)
        minus_one_indices = np.where(layers_np != 1)[0]
        mid_layers = layers_np[minus_one_indices].tolist()

        if len(mid_layers) > default.NN_MAX_DEPTH:
            raise Exception(f'[ERROR] The network cannot be deeper than {default.NN_MAX_DEPTH} layers')

        if np.any(mid_layers) > default.NN_MAX_WIDTH:
            raise Exception(f'[ERROR] Number of neurons in any layer cannot be more than {default.NN_MAX_WIDTH}')

        return mid_layers

    def generate_valid_isfet_mask_1d(self, chip_size=default.CHIP_SIZE_AVERAGED, dump_wells=''):

        """
        Returns ad 2D boolean array with all valid isfets on the chip. It excludes test and temperature isfets.

        :param chip_size: Tuple with chip dimensions in format (rows, columns)
        :return: Numpy boolean array (chip_rows*chip_columns, )
        """

        # Original valid mask
        mask_1d = indexing.generate_valid_isfet_mask(chip_size)

        # Dump wells mask
        if chip_size == default.CHIP_SIZE_FULL_RESOLUTION:
            if dump_wells == default.DUMP_WELLS_TOP:
                row_range = default.DUMP_WELLS_TOP_FULL_ROW_RANGE
                col_range = default.DUMP_WELLS_TOP_FULL_COL_RANGE
            elif dump_wells == default.DUMP_WELLS_LEFT:
                row_range = default.DUMP_WELLS_LEFT_FULL_ROW_RANGE
                col_range = default.DUMP_WELLS_LEFT_FULL_COL_RANGE
            elif dump_wells == '' or dump_wells is 'none' or dump_wells is None:
                row_range = None
                col_range = None
            else:
                raise Exception(f"[ERROR] Dump wells position '{dump_wells}' is not supported")
        elif chip_size == default.CHIP_SIZE_AVERAGED:
            if dump_wells == default.DUMP_WELLS_TOP:
                row_range = default.DUMP_WELLS_TOP_AVG_ROW_RANGE
                col_range = default.DUMP_WELLS_TOP_AVG_COL_RANGE
            elif dump_wells == default.DUMP_WELLS_LEFT:
                row_range = default.DUMP_WELLS_LEFT_AVG_ROW_RANGE
                col_range = default.DUMP_WELLS_LEFT_AVG_COL_RANGE
            elif dump_wells == '' or dump_wells is 'none' or dump_wells is None:
                row_range = None
                col_range = None
            else:
                raise Exception(f"[ERROR] Dump wells position '{dump_wells}' is not supported")
        else:
            raise Exception(f"[ERROR] Chip size {chip_size} is not supported")

        # Apply dump wells, if specified
        if row_range is not None:
            no_dump_wells_mask_2d = np.ones(chip_size, dtype=np.bool)
            no_dump_wells_mask_2d[row_range[0]:row_range[1], col_range[0]:col_range[1]] = False
            no_dump_wells_mask_1d = np.reshape(no_dump_wells_mask_2d, (no_dump_wells_mask_2d.size,))
            mask_1d = np.logical_and(mask_1d, no_dump_wells_mask_1d)

        return mask_1d