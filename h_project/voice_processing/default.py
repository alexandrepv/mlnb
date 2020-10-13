# Global references
SAMPLING_FREQUENCY = 44100

# Audio segmenter
SPECTOGRAM_THRESH = 0.0012 / 1000  # original 0.0012 / 750
WINDOW_SIZE = 512
GAP_KERNEL_SIZE = 3  # Gap = [T, F, T], invalid window = [F, T, F]
LABEL_DATAFRAME_KEY_LABEL = 'label'
LABEL_DATAFRAME_KEY_START_FRAME = 'start_frame'
LABEL_DATAFRAME_KEY_STOP_FRAME = 'stop_frame'

# Process AVG
#WINDOW_LENGTH = 128
THRESHOLD_TOP = 0.0007
THRESHOLD_BOTTOM = 0.0005
MIN_VALID_INTERVAL_LENGTH = 500  # Sample

# Voice and Silence Model
WINDOW_LENGTH = 512
WINDOW_STEP = 128
NUM_ADJACENT_WINDOWS = 2
NUM_MFCC_FEATURES = 13
SMOOTHING_AVG_WINDOW_LEN = 100

# Audio file extensions
supported_valid_extensions = ['.ogg', '.wav']
