# Isfet Classifier v3
ICV3_SELECTED_FLOWS = [1, 2, 3, 4]

# Isfet labelling for training
ISFET_LABEL_NO_SIGNAL = 0
ISFET_LABEL_NO_SIGNAL_NEXT_TO_VALID_SIGNAL = 1
ISFET_LABEL_AIR_BUBBLE = 2
ISFET_LABEL_VALID_SIGNAL_1MER = 10
ISFET_LABEL_VALID_SIGNAL_2MER = 11
ISFET_LABEL_VALID_SIGNAL_3MER = 12
ISFET_LABEL_VALID_SIGNAL_4MER = 13
ISFET_LABEL_VALID_SIGNAL_5MER = 14
ISFET_LABEL_VALID_SIGNAL_6MER = 15
ISFET_LABEL_VALID_SIGNAL_7MER = 16
ISFET_LABEL_VALID_SIGNAL_8MER = 17
ISFET_LABEL_VALID_SIGNAL_9MER = 18
ISFET_LABEL_VALID_SIGNAL_10MER = 19


# Signal Extraction
SIGNAL_EXTR_NUM_SAMPLES_PER_FLOW = 196  # About 10 seconds of data per flow with sampling rate of 19.53 Hz
SIGNAL_EXTR_NUM_FLOWS = 8
SIGNAL_EXTR_NUM_CLUSTERS_FOR_NEGATIVE_SAMPLING = 5
SIGNAL_EXTR_NEGATIVE_SAMPLING_RANDOM_STATE = 42
SIGNAL_EXTR_GLOBAL_REF_PERCENTILE = 10.0  # %
SIGNAL_EXTR_LPF_CUTOFF_FREQ = 0.7  # Hz
SIGNAL_EXTR_AIR_BUBBLE_THRESHOLD = 0.62
SIGNAL_EXTR_MAX_ISFETS_PER_DATA_SLICE = 15000
SIGNAL_EXTR_NUM_ISFETS_FOR_ZEROING = 10
SIGNAL_EXTR_NORM_MIN_VALUE = -2.0  # mV
SIGNAL_EXTR_NORM_MAX_VALUE = 15.0  # mV
SIGNAL_EXTR_NEGATIVE_LABEL = 0
SIGNAL_EXTR_POSITIVE_LABEL = 1

# Labelled Signals H5 file
H5_SIGNALS_KEY_RUNID = 'runid'
H5_SIGNALS_KEY_CHIP_ROWS = 'chip_rows'
H5_SIGNALS_KEY_CHIP_COLS = 'chip_cols'
H5_SIGNALS_KEY_NUM_FLOWS = 'num_flows'
H5_SIGNALS_KEY_NUM_SAMPLES_PER_FLOW = 'num_samples_per_flow'
H5_SIGNALS_KEY_LFP_CUTOFF_FREQ = 'lpf_cutoff_freq'
H5_SIGNALS_KEY_AIR_BUBBLE_THRESHOLD = 'air_bubble_threshold'
H5_SIGNALS_KEY_REF_PERCENTILE = 'ref_percentile'

H5_SIGNALS_GROUP_KEY_FLOW_INDEX = 'flow_index'
H5_SIGNALS_GROUP_KEY_SIGNALS = 'signals'
H5_SIGNALS_GROUP_KEY_LABELS = 'labels'
H5_SIGNALS_GROUP_KEY_INDICES = 'indices'
H5_SIGNALS_GROUP_KEY_REF_INDICES = 'ref_indices'

SHUFFLING_DATA_RANDOM_STATE = 42
FLATFILE_DEFAULT_REF_ID = 999

# Neural Network
NN_MID_LAYERS_ACTIVATION_FUNCTION = 'relu'
NN_FINAL_LAYER_ACTIVATION_FUNCTION = 'sigmoid'
NN_LOSS_FUNCTION = 'binary_crossentropy'
NN_METRICS = ['accuracy']
NN_OPTIMIZER = 'adam'
NN_LEARNING_RATE = 0.01
NN_EPOCHS = 30
NN_MAX_WIDTH = 1024  # Maximum number of neurons at any layer
NN_MAX_DEPTH = 8  # Maximum number of layers
NN_LAYERS = [200, 25]  # Middle layers of the network
NN_EARLY_STOPPING_PATIENCE = 5  # Wait for 5 unchanged epochs before terminating the training
NN_VALIDATION_SPLIT = 0.1

NN_TRAINING_REPORT_KEY_LAYERS = 'layers'
NN_TRAINING_REPORT_KEY_INPUT_DIMENSIONS = 'input_dimensions'
NN_TRAINING_REPORT_KEY_OPTIMIZER = 'optimizer'
NN_TRAINING_REPORT_KEY_LR = 'learning_rate'
NN_TRAINING_REPORT_KEY_EPOCHS = 'epochs'
NN_TRAINING_REPORT_KEY_LOSS = 'loss'
NN_TRAINING_REPORT_KEY_ACCURACY = 'accuracy'

NN_VALIDATOR_TEST_SIZE = 0.2
NN_VALIDATOR_RANDOM_STATE = 42
NN_VALIDATOR_LAYER_SIZES = [25, 50, 75, 100, 200, 300, 400]
NN_VALIDATOR_MAX_LAYERS = 3
NN_VALIDATOR_EPOCHS = 5
NN_VALIDATOR_MAX_DECIMAL_PLACES = 4
NN_VALIDATOR_NUM_STEPS_PER_DECIMAL = 6
NN_VALIDATOR_MODES = ['layer_combinations', 'learning_rates']
NN_VALIDATOR_REPORT_KEY_LAYERS = 'layers'
NN_VALIDATOR_REPORT_KEY_OPTIMIZER = 'optimizer'
NN_VALIDATOR_REPORT_KEY_LEARNING_RATE = 'learning_rate'
NN_VALIDATOR_REPORT_KEY_EPOCHS = 'epochs'
NN_VALIDATOR_REPORT_KEY_LOSS = 'loss'
NN_VALIDATOR_REPORT_KEY_ACCURACY = 'accuracy'
NN_VALIDATOR_REPORT_KEY_VAL_LOSS = 'val_loss'
NN_VALIDATOR_REPORT_KEY_VAL_ACCURACY = 'val_accuracy'
NN_VALIDATOR_REPORT_COLUMNS = [NN_VALIDATOR_REPORT_KEY_LAYERS,
                               NN_VALIDATOR_REPORT_KEY_OPTIMIZER,
                               NN_VALIDATOR_REPORT_KEY_LEARNING_RATE,
                               NN_VALIDATOR_REPORT_KEY_EPOCHS,
                               NN_VALIDATOR_REPORT_KEY_LOSS,
                               NN_VALIDATOR_REPORT_KEY_ACCURACY]

NN_RECIPE_KEY_MODEL_H5_FILENAME = 'model_h5_filename'
NN_RECIPE_KEY_LAYERS = 'layers'
NN_RECIPE_KEY_OPTIMIZER = 'optimizer'
NN_RECIPE_KEY_LEARNING_RATE = 'learning_rate'
NN_RECIPE_KEY_NORM_MIN_VALUE = 'norm_min_value'
NN_RECIPE_KEY_NORM_MAX_VALUE = 'norm_max_value'
NN_RECIPE_KEY_INPUT_DATA_DIMENSIONS = 'input_data_dimensions'
NN_RECIPE_KEY_REF_PERCENTILE = 'ref_percentile'

LOGO_LINES = ['','','','','','']
LOGO_LINES[0] = "  _____      __     _      _____ _                  __                     ____ "
LOGO_LINES[1] = " |_   _|    / _|   | |    / ____| |             (_)/ _(_)                 |___ \\"
LOGO_LINES[2] = "   | |  ___| |_ ___| |_  | |    | | __ _ ___ ___ _| |_ _  ___ _ __  __   __ __) |"
LOGO_LINES[3] = "   | | / __|  _/ _ \ __| | |    | |/ _` / __/ __| |  _| |/ _ \ '__| \ \ / /|__ <"
LOGO_LINES[4] = "  _| |_\__ \ ||  __/ |_  | |____| | (_| \__ \__ \ | | | |  __/ |     \ V / ___) |"
LOGO_LINES[5] = " |_____|___/_| \___|\__|  \_____|_|\__,_|___/___/_|_| |_|\___|_|      \_/ |____/"

LOGO_NAME = '\n'.join(LOGO_LINES)

PLOT_TITLE_SIZE = 18
PLOT_AXES_LABEL_SIZE = 15

# Chip dimensions
CHIP_SIZE_AVERAGED = (800, 1920)
CHIP_SIZE_FULL_RESOLUTION = (CHIP_SIZE_AVERAGED[0] * 2, CHIP_SIZE_AVERAGED[1] * 2)

# Dump wells
DUMP_WELLS_TOP_AVG_ROW_RANGE = (0, 70)
DUMP_WELLS_TOP_AVG_COL_RANGE = (0, CHIP_SIZE_AVERAGED[1])
DUMP_WELLS_LEFT_AVG_ROW_RANGE = (0, CHIP_SIZE_AVERAGED[0])
DUMP_WELLS_LEFT_AVG_COL_RANGE = (0, 64)
DUMP_WELLS_TOP_FULL_ROW_RANGE = (0, DUMP_WELLS_TOP_AVG_ROW_RANGE[1] * 2)
DUMP_WELLS_TOP_FULL_COL_RANGE = (0, CHIP_SIZE_FULL_RESOLUTION[1])
DUMP_WELLS_LEFT_FULL_ROW_RANGE = (0, CHIP_SIZE_FULL_RESOLUTION[0])
DUMP_WELLS_LEFT_FULL_COL_RANGE = (0, DUMP_WELLS_LEFT_AVG_COL_RANGE[1] * 2)
DUMP_WELLS_NONE = 'none'
DUMP_WELLS_TOP = 'top'
DUMP_WELLS_LEFT = 'left'
DUMP_WELLS_LOCATIONS = [DUMP_WELLS_NONE, DUMP_WELLS_TOP, DUMP_WELLS_LEFT]