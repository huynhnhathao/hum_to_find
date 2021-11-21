TRAIN_ANNOTATIONS_FILE = "/home/huynhhao/Desktop/hum/hum_to_find/meta_data/train_annotation.csv"
VAL_ANNOTATION_FILE = '/home/huynhhao/Desktop/hum/hum_to_find/meta_data/val_annotation.csv'
TRAIN_AUDIO_DIR = "/home/huynhhao/Desktop/hum/data"
VAL_AUDIO_DIR = ""

SECS = 10
SAMPLE_RATE = 16000
NUM_SAMPLES = 160000 


BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIMS = 128
# each (song, hum) will has 8 triplet, 16*8 = 128 triplet each batch
NUM_SONG_EACH_BATCH = 16
# the threshold to consider if someone is singing
SINGING_THRESHOLD = 0.1
DEVICE = 'cpu'

# transformer parameters
TRANSFORMER_NFFT = 1024
TRANSFORMER_HOP_LENGTH = 512
N_MELS = 64
NUM_CHUNKS_EACH_AUDIO = 9