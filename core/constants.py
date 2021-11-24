TRAIN_ANNOTATIONS_FILE = "/home/huynhhao/Desktop/hum/hum_to_find/meta_data/train_annotation.csv"
VAL_ANNOTATION_FILE = '/home/huynhhao/Desktop/hum/hum_to_find/meta_data/val_annotation.csv'
TRAIN_AUDIO_DIR = "/home/huynhhao/Desktop/hum/data"
VAL_AUDIO_DIR = '/home/huynhhao/Desktop/hum/data/'
SAVE_EMBEDDING_PATH = '/home/huynhhao/Desktop/hum/data/cached_embeddings'
SAVE_VAL_FEATURES_PATH = "/home/huynhhao/Desktop/hum/data/cached_features/val/val_spectrogram_nosplit.pt"
SAVE_TRAIN_FEATURES_PATH = "/home/huynhhao/Desktop/hum/data/cached_features/train/train_spectrogram_nosplit.pt"
SAVE_MODEL_PATH = '/home/huynhhao/Desktop/hum'
LOG_FILE_PATH = '/home/huynhhao/Desktop/hum/hum_to_find/core/log.txt'


SECS = 10
SAMPLE_RATE = 16000
NUM_SAMPLES = 160000 


BATCH_SIZE = 256
EPOCHS = 100
CHECKPOINT_EPOCHS = 5
EVAL_EACH_NUM_EPOCHS = 1

LEARNING_RATE = 0.001
EMBEDDING_DIMS = 512
# each (song, hum) will has 8 triplet, 16*8 = 128 triplet each batch
NUM_SONG_EACH_BATCH = 16
# the threshold to consider if someone is singing
SINGING_THRESHOLD = 0.1
DEVICE = 'cpu'

# transformer parameters
TRANSFORMER_NFFT = 1024
TRANSFORMER_HOP_LENGTH = 512
N_MELS = 96

# The threshold in euclidean distance, lower than this threshold will be a match.
MATCHED_THRESHOLD = 1.1

NUM_SONG_RETRIEVED_PER_QUERY = 10